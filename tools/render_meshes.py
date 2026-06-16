#!/usr/bin/env python
"""Prototype mesh -> composite-PNG renderer for proofreader-mcp.

Turns per-object meshes into the one-composite-PNG-per-object input the MCP
server consumes, plus an optional ``objects.json`` sidecar. Each PNG is a 2x2
montage of fixed viewpoints with a neutral background.

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
import json
import sys
from pathlib import Path

from mesh_sources import MeshSource, make_source

# Six axis-aligned viewpoints as (label, camera-direction, up). The camera sits
# along each +/- axis looking back at the object center, so the montage shows the
# object from +X, -X, +Y, -Y, +Z, -Z. Consistent across objects.
VIEWS: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = [
    ("+X", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("-X", (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("+Y", (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ("-Y", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    ("+Z", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ("-Z", (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
]
GRID_ROWS, GRID_COLS = 2, 3


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
) -> tuple[float, float, float]:
    plotter = pv.Plotter(
        off_screen=True,
        shape=(GRID_ROWS, GRID_COLS),
        window_size=(cell_size * GRID_COLS, cell_size * GRID_ROWS),
    )
    plotter.set_background(background)
    center = mesh.center
    dx, dy, dz = _extents_nm(mesh)
    extent_text = f"{dx:.0f} x {dy:.0f} x {dz:.0f} nm"
    # Camera distance: large enough to clear the object; irrelevant to scale under
    # parallel projection, which is fixed by parallel_scale below.
    half = max(dx, dy, dz) / 2.0
    distance = (parallel_scale or half) * 4.0 + half

    for i, (label, direction, up) in enumerate(VIEWS):
        plotter.subplot(i // GRID_COLS, i % GRID_COLS)
        plotter.add_mesh(mesh, color=color, smooth_shading=True, specular=0.3)
        plotter.add_text(label, position="upper_left", font_size=12, color="black")
        plotter.add_text(
            extent_text, position="lower_left", font_size=10, color="dimgray"
        )
        position = [center[j] + direction[j] * distance for j in range(3)]
        plotter.camera_position = [tuple(position), tuple(center), up]
        if parallel_scale is not None:
            # Set last: orientation changes can reset the parallel scale.
            plotter.enable_parallel_projection()
            plotter.camera.parallel_scale = parallel_scale
        else:
            plotter.reset_camera()

    plotter.screenshot(str(out_path))
    plotter.close()
    return (dx, dy, dz)


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
        "--cell-size", type=int, default=512, help="pixel size of each montage panel"
    )
    parser.add_argument("--color", default="lightgray", help="mesh color")
    parser.add_argument("--background", default="white", help="neutral background")
    parser.add_argument(
        "--scale-mode",
        choices=["global", "per-object"],
        default="global",
        help="global keeps absolute size comparable across objects (recommended)",
    )
    parser.add_argument("--limit", type=int, default=None, help="render only first N")
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

    pv.OFF_SCREEN = True

    try:
        source = make_source(
            args.input, kind=args.source, pattern=args.pattern, lod=args.lod
        )
    except Exception as exc:  # noqa: BLE001 - surface any backend setup error cleanly
        print(f"could not open source: {exc}", file=sys.stderr)
        return 2

    try:
        ids = _select_ids(source, args.ids, args.limit)
        if not ids:
            print("no objects to render", file=sys.stderr)
            return 1

        args.output_dir.mkdir(parents=True, exist_ok=True)
        collection_id = args.collection_id or Path(args.input).resolve().name

        parallel_scale = None
        if args.scale_mode == "global":
            parallel_scale = _global_half_extent(source, ids) * 1.1  # small margin

        objects = []
        for oid in tqdm(ids, desc="rendering", unit="obj"):
            out_path = args.output_dir / f"{oid}.png"
            dx, dy, dz = _render_one(
                pv,
                source.get(oid),
                out_path,
                cell_size=args.cell_size,
                color=args.color,
                background=args.background,
                parallel_scale=parallel_scale,
            )
            objects.append(
                {
                    "object_id": oid,
                    "image": out_path.name,
                    "meta": {"extent_nm": [round(dx, 1), round(dy, 1), round(dz, 1)]},
                }
            )
    finally:
        source.close()

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
