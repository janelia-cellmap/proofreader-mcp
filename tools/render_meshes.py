#!/usr/bin/env python
"""Prototype mesh -> composite-PNG renderer for proofreader-mcp.

Turns per-object meshes into the one-composite-PNG-per-object input the MCP
server consumes, plus an optional ``objects.json`` sidecar. Each PNG is a 2x3
montage of the six axis-aligned views (columns = x/y/z, + on top, - beneath) on a
neutral background, with each axis's extent in nm. Rendering is parallelized across
processes (``--workers``, default = available CPUs), since VTK is not thread-safe.

Two input sources (auto-detected, or forced with ``--source``):

* **files** — a directory of ``{object_id}.ply`` (or .obj/.stl) meshes.
* **precomputed** — a Neuroglancer precomputed mesh layer (sharded/unsharded,
  multi-LOD Draco), local or remote (``gs://``, ``https://``). Pick the level of
  detail with ``--lod`` (0 = finest; higher = coarser/faster).

This is a PROTOTYPE living outside the installed package (see the ``[render]`` and
``[precomputed]`` optional extras). It is intended to migrate into cellmap-analyze,
where dask can scale it and segmentation access can enrich the sidecar
(voxel_count, at_volume_edge, centroid). Run it with::

    uv sync --extra render --extra precomputed

    # PLY files:
    uv run python tools/render_meshes.py /path/to/plys /path/to/out \
        --organelle-type mitochondrion

    # precomputed multires at LOD 1:
    uv run python tools/render_meshes.py /path/to/mesh/multires /path/to/out \
        --source precomputed --lod 1 --organelle-type mitochondrion --limit 50

Headless note: VTK needs an offscreen GL context. On a cluster use a VTK build
with OSMesa/EGL, or wrap the call in ``xvfb-run``.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from mesh_sources import MeshSource, make_source


def _available_cpus() -> int:
    """CPU count to parallelize over, cross-platform.

    ``os.sched_getaffinity`` (Linux) respects cpu affinity / cgroup limits; macOS
    and others fall back to ``os.cpu_count``.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# Six axis-aligned viewpoints as (label, camera-direction, up, axis). The camera
# sits along each +/- axis looking back at the object center. Laid out as a 2x3
# grid so each axis is a COLUMN with the + view on top and the - view beneath it:
#   +X +Y +Z
#   -X -Y -Z
# ``axis`` (0/1/2 for x/y/z, or None) marks which panel shows that axis's extent,
# so each column is annotated once.
VIEWS: list[
    tuple[str, tuple[float, float, float], tuple[float, float, float], int | None]
] = [
    ("+X", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0),
    ("+Y", (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 1),
    ("+Z", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), 2),
    ("-X", (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), None),
    ("-Y", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), None),
    ("-Z", (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), None),
]
GRID_ROWS, GRID_COLS = 2, 3
AXIS_NAMES = ("x", "y", "z")


def _global_half_extent(source: MeshSource, ids: list[str]) -> float:
    """Largest half-extent across the selected meshes, for a shared scale.

    Rendering every object at the same world scale keeps size comparable by eye,
    which is what makes 'too small => junk' judgments possible without metadata.
    Note: this loads each mesh once here and again at render time.
    """
    half = 0.0
    for oid in ids:
        b = source.get(oid).bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        span = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
        half = max(half, span / 2.0)
    return half or 1.0


def _extents_nm(mesh) -> tuple[float, float, float]:
    """(dx, dy, dz) bounding-box extents. Mesh coordinates are already in nm."""
    b = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    return (b[1] - b[0], b[3] - b[2], b[5] - b[4])


def _render_one(
    pv,
    mesh,
    out_path: Path,
    cell_size: int,
    color: str,
    background: str,
    parallel_scale: float | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    plotter = pv.Plotter(
        off_screen=True,
        shape=(GRID_ROWS, GRID_COLS),
        window_size=(cell_size * GRID_COLS, cell_size * GRID_ROWS),
    )
    plotter.set_background(background)
    center = mesh.center
    extents = _extents_nm(mesh)
    # Camera distance: large enough to clear the object; irrelevant to scale under
    # parallel projection, which is fixed by parallel_scale below.
    half = max(extents) / 2.0
    distance = (parallel_scale or half) * 4.0 + half
    # Font sizes scale with the panel so labels stay large and legible while still
    # fitting the full "x_extent = NNNN nm" string within one panel's width.
    axis_font = max(14, cell_size // 20)
    extent_font = max(13, cell_size // 28)

    for i, (label, direction, up, axis) in enumerate(VIEWS):
        plotter.subplot(i // GRID_COLS, i % GRID_COLS)
        plotter.add_mesh(mesh, color=color, smooth_shading=True, specular=0.3)
        plotter.add_text(
            label, position="upper_left", font_size=axis_font, color="black"
        )
        if axis is not None:
            plotter.add_text(
                f"{AXIS_NAMES[axis]}_extent = {extents[axis]:.0f} nm",
                position="lower_left",
                font_size=extent_font,
                color="black",
            )
        position = [center[j] + direction[j] * distance for j in range(3)]
        plotter.camera_position = [tuple(position), tuple(center), up]
        plotter.enable_parallel_projection()
        if parallel_scale is not None:
            # Shared scale: set last, as orientation changes can reset it.
            plotter.camera.parallel_scale = parallel_scale
        else:
            # Per-object: fit this view's in-plane extents (the two axes that are
            # NOT the viewing axis) so the object fills the square panel without
            # clipping. parallel_scale is half the panel's world height.
            up_axis = max(range(3), key=lambda k: abs(up[k]))
            view_axis = max(range(3), key=lambda k: abs(direction[k]))
            horiz_axis = 3 - up_axis - view_axis
            plotter.camera.parallel_scale = (
                max(extents[up_axis], extents[horiz_axis]) / 2.0 * 1.1
            )

    plotter.screenshot(str(out_path))
    plotter.close()
    return extents, (float(center[0]), float(center[1]), float(center[2]))


def _select_ids(
    source: MeshSource, ids_arg: str | None, limit: int | None
) -> list[str]:
    if ids_arg:
        # comma-separated list, or a path to a newline-delimited file
        p = Path(ids_arg)
        raw = p.read_text().split() if p.exists() else ids_arg.split(",")
        ids = [s.strip() for s in raw if s.strip()]
    else:
        ids = source.list_ids()
    return ids[:limit] if limit is not None else ids


# --------------------------------------------------------------------------
# Process-pool workers. pyvista/VTK is not thread-safe, so we render in separate
# processes. Each worker builds its own MeshSource once (CloudVolume objects and
# VTK contexts don't cross the process boundary).
# --------------------------------------------------------------------------

_WORKER: dict = {}


def _init_worker(kind: str, source: str, pattern: str, lod: int) -> None:
    import pyvista as pv

    pv.OFF_SCREEN = True
    src = make_source(source, kind=kind, pattern=pattern, lod=lod)
    _WORKER["source"] = src
    _WORKER["pv"] = pv
    atexit.register(src.close)


def _worker_bounds(oid: str) -> tuple[str, tuple[float, float, float]]:
    return oid, _extents_nm(_WORKER["source"].get(oid))


def _worker_render(task: tuple):
    oid, out_dir, cell_size, color, background, parallel_scale = task
    ext, center = _render_one(
        _WORKER["pv"],
        _WORKER["source"].get(oid),
        Path(out_dir) / f"{oid}.png",
        cell_size=cell_size,
        color=color,
        background=background,
        parallel_scale=parallel_scale,
    )
    return oid, ext, center


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="mesh dir, or precomputed layer path/URL")
    parser.add_argument("output_dir", type=Path, help="dir to write {object_id}.png")
    parser.add_argument(
        "--source",
        choices=["auto", "files", "precomputed"],
        default="auto",
        help="input type (auto-detects precomputed by info/URL)",
    )
    parser.add_argument("--pattern", default="*.ply", help="glob for file source")
    parser.add_argument(
        "--lod",
        type=int,
        default=0,
        help="precomputed level of detail (0=finest; falls back if unavailable)",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="comma-separated ids or path to a newline-delimited id file",
    )
    parser.add_argument("--organelle-type", default=None, help="for sidecar + rubric")
    parser.add_argument("--collection-id", default=None, help="defaults to input name")
    parser.add_argument(
        "--cell-size", type=int, default=640, help="pixel size of each montage panel"
    )
    parser.add_argument("--color", default="lightgray", help="mesh color")
    parser.add_argument("--background", default="white", help="neutral background")
    parser.add_argument(
        "--scale-mode",
        choices=["per-object", "global"],
        default="per-object",
        help="per-object fills the frame with each object (default); "
        "global uses one shared scale so sizes are comparable by eye",
    )
    parser.add_argument("--limit", type=int, default=None, help="render only first N")
    parser.add_argument(
        "--workers",
        type=int,
        default=_available_cpus(),
        help="parallel render processes (default: available CPUs)",
    )
    parser.add_argument(
        "--no-sidecar", action="store_true", help="skip writing objects.json"
    )
    args = parser.parse_args(argv)

    try:
        import pyvista as pv
    except ImportError:
        print("pyvista not installed. Run: uv sync --extra render", file=sys.stderr)
        return 2

    from tqdm import tqdm

    from mesh_sources import is_precomputed

    pv.OFF_SCREEN = True

    # Resolve the source kind once so each worker doesn't re-detect it.
    kind = args.source
    if kind == "auto":
        kind = "precomputed" if is_precomputed(args.input) else "files"

    # Enumerate ids (and validate the source opens) in the main process.
    try:
        index = make_source(args.input, kind=kind, pattern=args.pattern, lod=args.lod)
    except Exception as exc:  # noqa: BLE001 - surface any backend setup error cleanly
        print(f"could not open source: {exc}", file=sys.stderr)
        return 2
    try:
        ids = _select_ids(index, args.ids, args.limit)
    finally:
        index.close()
    if not ids:
        print("no objects to render", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    collection_id = args.collection_id or Path(args.input).resolve().name
    workers = max(1, min(args.workers, len(ids)))
    use_global = args.scale_mode == "global"

    extents: dict[str, tuple[float, float, float]] = {}
    centers: dict[str, tuple[float, float, float]] = {}
    parallel_scale: float | None = None

    if workers == 1:
        source = make_source(args.input, kind=kind, pattern=args.pattern, lod=args.lod)
        try:
            if use_global:
                parallel_scale = _global_half_extent(source, ids) * 1.1
            for oid in tqdm(ids, desc="rendering", unit="obj"):
                extents[oid], centers[oid] = _render_one(
                    pv,
                    source.get(oid),
                    args.output_dir / f"{oid}.png",
                    cell_size=args.cell_size,
                    color=args.color,
                    background=args.background,
                    parallel_scale=parallel_scale,
                )
        finally:
            source.close()
    else:
        init = (kind, args.input, args.pattern, args.lod)
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=init
        ) as ex:
            # Global scale needs every object's bounds before any render starts.
            if use_global:
                half = 0.0
                for oid, ext in tqdm(
                    ex.map(_worker_bounds, ids),
                    total=len(ids),
                    desc="bounds",
                    unit="obj",
                ):
                    extents[oid] = ext
                    half = max(half, max(ext) / 2.0)
                parallel_scale = (half or 1.0) * 1.1
            tasks = [
                (
                    oid,
                    str(args.output_dir),
                    args.cell_size,
                    args.color,
                    args.background,
                    parallel_scale,
                )
                for oid in ids
            ]
            for oid, ext, ctr in tqdm(
                ex.map(_worker_render, tasks),
                total=len(ids),
                desc="rendering",
                unit="obj",
            ):
                extents[oid] = ext
                centers[oid] = ctr

    objects = [
        {
            "object_id": oid,
            "image": f"{oid}.png",
            "meta": {
                "extent_nm": [round(v, 1) for v in extents[oid]],
                "centroid_nm": [round(v, 1) for v in centers[oid]],
            },
        }
        for oid in ids
    ]

    if not args.no_sidecar:
        sidecar = {
            "collection_id": collection_id,
            "organelle_type": args.organelle_type,
            "objects": objects,
        }
        (args.output_dir / "objects.json").write_text(json.dumps(sidecar, indent=2))

    print(f"rendered {len(objects)} objects -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
