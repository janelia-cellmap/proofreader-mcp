"""Mesh-source abstraction for the prototype renderer.

A source yields ``(object_id, pyvista.PolyData)`` pairs from one of:

* **files** — a directory of per-object mesh files (``{object_id}.ply`` etc.),
  read by pyvista.
* **precomputed** — a Neuroglancer precomputed mesh layer (sharded or unsharded,
  multi-LOD Draco), local or remote (``gs://``, ``https://``), read by
  CloudVolume at a chosen level of detail.

Heavy imports (pyvista, cloudvolume, numpy) are deferred to call time so the
module imports cheaply and each backend is only required when actually used.
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

MESH_INFO_TYPES = {"neuroglancer_multilod_draco", "neuroglancer_legacy_mesh"}


def _strip_scheme(url: str) -> str:
    """Drop a leading ``precomputed://`` and any trailing slash."""
    if url.startswith("precomputed://"):
        url = url[len("precomputed://") :]
    return url.rstrip("/")


def _read_remote_json(url: str, suffix: str) -> dict | None:
    """GET ``<url>/<suffix>`` and parse JSON, or None on any failure."""
    try:
        with urllib.request.urlopen(f"{_strip_scheme(url)}/{suffix}", timeout=15) as r:
            return json.loads(r.read().decode())
    except Exception:  # noqa: BLE001 - missing/unreachable info is a normal "no"
        return None


class MeshSource(ABC):
    """Yields object ids and their meshes as pyvista PolyData."""

    @abstractmethod
    def list_ids(self) -> list[str]: ...

    @abstractmethod
    def get(self, object_id: str):  # -> pyvista.PolyData
        ...

    def close(self) -> None:  # noqa: B027 - optional no-op cleanup hook
        pass


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------


class FileMeshSource(MeshSource):
    """A directory of pyvista-readable mesh files, one per object."""

    def __init__(self, input_dir: Path, pattern: str = "*.ply") -> None:
        self.input_dir = Path(input_dir)
        self.pattern = pattern
        self._paths = {p.stem: p for p in sorted(self.input_dir.glob(pattern))}

    def list_ids(self) -> list[str]:
        return list(self._paths)

    def get(self, object_id: str):
        import pyvista as pv

        return pv.read(self._paths[object_id])


# --------------------------------------------------------------------------
# precomputed (Neuroglancer)
# --------------------------------------------------------------------------


def _read_info(base: Path) -> dict | None:
    info = base / "info"
    if not info.exists():
        return None
    try:
        return json.loads(info.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def is_precomputed(source: str) -> bool:
    """True if ``source`` looks like a precomputed layer (URL or local dir)."""
    if "://" in source:
        return True
    base = Path(source)
    info = _read_info(base)
    if info is None:
        return False
    atype = info.get("@type", "")
    return atype in MESH_INFO_TYPES or info.get("type") == "segmentation"


class PrecomputedMeshSource(MeshSource):
    """Read a precomputed mesh layer via CloudVolume at a fixed LOD.

    Accepts a segmentation layer (with a ``mesh`` key) or a standalone mesh
    layer directory (its own ``info`` of type ``neuroglancer_*`` mesh). For a
    standalone local mesh dir, a minimal parent segmentation layer is synthesized
    in a temp dir so CloudVolume can attach the mesh source.
    """

    def __init__(self, source: str, lod: int = 0) -> None:
        self.lod = lod
        self._tmp: str | None = None
        cloudpath, self._segprop_ids, info = self._resolve(source)

        from cloudvolume import CloudVolume

        # ``info`` is set when we synthesize a parent segmentation for a
        # standalone mesh layer; passing it inline stops CloudVolume from trying
        # to fetch a (nonexistent) segmentation ``info`` for that layer.
        self.cv = CloudVolume(cloudpath, progress=False, info=info)

    def _resolve(self, source: str) -> tuple[str, list[str] | None, dict | None]:
        """Return a CloudVolume cloudpath, inline segment ids, and inline info.

        ``info`` is non-None only when ``source`` is a *standalone mesh layer*
        (its own ``info`` is a ``neuroglancer_*`` mesh, not a segmentation), in
        which case we synthesize a minimal parent segmentation so CloudVolume can
        attach the mesh. This handles both local dirs (via a temp symlink) and
        remote URLs (via the parent URL + an inline info).
        """
        if "://" in source:
            info = _read_remote_json(source, "info")
            if info is not None and info.get("@type") in MESH_INFO_TYPES:
                # Standalone *remote* mesh layer: treat its parent dir as the
                # segmentation and name this dir as the mesh subdir.
                base = _strip_scheme(source)
                parent, mesh_dir = base.rsplit("/", 1)
                ids = self._remote_segment_ids(source)
                return f"precomputed://{parent}/", ids, self._mesh_seg_info(mesh_dir)
            cp = (
                source
                if source.startswith("precomputed://")
                else f"precomputed://{source}"
            )
            return cp, None, None

        base = Path(source).resolve()
        info = _read_info(base)
        if info is None:
            raise ValueError(f"no precomputed 'info' found at {base}")

        # A standalone mesh layer: synthesize a parent segmentation layer in a
        # temp dir whose ``mesh`` subdir symlinks the real local mesh dir.
        if info.get("@type") in MESH_INFO_TYPES:
            ids = self._segment_property_ids(base)
            self._tmp = tempfile.mkdtemp(prefix="proofreader_pc_")
            info_path = Path(self._tmp) / "info"
            info_path.write_text(json.dumps(self._mesh_seg_info("mesh")))
            os.symlink(base, Path(self._tmp) / "mesh")
            return f"precomputed://file://{self._tmp}", ids, None

        # A full segmentation layer: use as-is; try its mesh segment_properties.
        mesh_key = info.get("mesh", "mesh")
        ids = self._segment_property_ids(base / mesh_key)
        return f"precomputed://file://{base}", ids, None

    @staticmethod
    def _mesh_seg_info(mesh_dir: str) -> dict:
        """Minimal parent-segmentation info pointing ``mesh`` at ``mesh_dir``."""
        return {
            "@type": "neuroglancer_multiscale_volume",
            "type": "segmentation",
            "data_type": "uint64",
            "num_channels": 1,
            "scales": [
                {
                    "key": "s0",
                    "size": [1, 1, 1],
                    "resolution": [1, 1, 1],
                    "chunk_sizes": [[64, 64, 64]],
                    "encoding": "raw",
                    "voxel_offset": [0, 0, 0],
                }
            ],
            "mesh": mesh_dir,
        }

    @staticmethod
    def _remote_segment_ids(url: str) -> list[str] | None:
        data = _read_remote_json(url, "segment_properties/info")
        try:
            return list(data["inline"]["ids"]) if data else None
        except (KeyError, TypeError):
            return None

    @staticmethod
    def _segment_property_ids(mesh_dir: Path) -> list[str] | None:
        sp = mesh_dir / "segment_properties" / "info"
        if not sp.exists():
            return None
        try:
            data = json.loads(sp.read_text())
            return list(data["inline"]["ids"])
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def list_ids(self) -> list[str]:
        if self._segprop_ids is None:
            raise ValueError(
                "no segment_properties found; pass --ids explicitly for this layer"
            )
        return self._segprop_ids

    def get(self, object_id: str):
        import numpy as np
        import pyvista as pv

        sid = int(object_id)
        mesh = self._get_at_best_lod(sid)
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        # pyvista face layout: [n, i0, i1, i2, n, ...]
        padded = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
        return pv.PolyData(verts, padded)

    def _get_at_best_lod(self, sid: int):
        """Fetch at the requested LOD, falling back to coarser if unavailable."""
        last_err: Exception | None = None
        for lod in range(self.lod, -1, -1):
            try:
                res = self.cv.mesh.get(sid, lod=lod)
                return res[sid] if isinstance(res, dict) else res
            except Exception as exc:  # noqa: BLE001 - CloudVolume raises various types
                last_err = exc
        raise RuntimeError(f"could not fetch mesh {sid}: {last_err}")

    def close(self) -> None:
        if self._tmp and os.path.isdir(self._tmp):
            link = Path(self._tmp) / "mesh"
            if link.is_symlink():
                link.unlink()
            for child in Path(self._tmp).glob("*"):
                if child.is_file():
                    child.unlink()
            os.rmdir(self._tmp)
            self._tmp = None


def make_source(
    source: str, kind: str = "auto", pattern: str = "*.ply", lod: int = 0
) -> MeshSource:
    """Build the right MeshSource. ``kind`` is auto/files/precomputed."""
    if kind == "auto":
        kind = "precomputed" if is_precomputed(source) else "files"
    if kind == "precomputed":
        return PrecomputedMeshSource(source, lod=lod)
    return FileMeshSource(Path(source), pattern=pattern)
