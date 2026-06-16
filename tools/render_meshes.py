#!/usr/bin/env python
"""Prototype mesh -> composite-PNG renderer for proofreader-mcp.

Turns a directory of per-object meshes ({object_id}.ply) into the one-composite-
PNG-per-object input the MCP server consumes, plus an optional ``objects.json``
sidecar. Each PNG is a 2x2 montage of fixed viewpoints with a neutral background.

This is a PROTOTYPE living outside the installed package (see the ``[render]``
optional extra). It is intended to migrate into cellmap-analyze, where dask and
segmentation access can scale it and enrich the sidecar (voxel_count,
at_volume_edge, centroid). Run it with::

    uv sync --extra render
    uv run python tools/render_meshes.py INPUT_DIR OUTPUT_DIR \
        --organelle-type mitochondrion

Headless note: VTK needs an offscreen GL context. On a cluster either use a build
with OSMesa (``pip install --no-deps vtk-osmesa`` over pyvista's vtk) or pass
``--xvfb`` to spin up a virtual framebuffer via ``pyvista.start_xvfb()``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Four fixed viewpoints as (label, azimuth, elevation) offsets from a front view.
# Consistent across objects so the host LLM sees the same orientations everywhere.
VIEWS: list[tuple[str, float, float]] = [
    ("front", 0.0, 0.0),
    ("side", 90.0, 0.0),
    ("top", 0.0, 90.0),
    ("perspective", 45.0, 30.0),
]


def _iter_meshes(input_dir: Path, pattern: str) -> list[Path]:
    return sorted(input_dir.glob(pattern))


def _global_half_extent(paths: list[Path], pv) -> float:
    """Largest half-extent across all meshes, for a shared parallel-projection scale.

    Rendering every object at the same world scale keeps size comparable by eye,
    which is what makes 'too small => junk' judgments possible without metadata.
    """
    half = 0.0
    for p in paths:
        b = pv.read(p).bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        span = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
        half = max(half, span / 2.0)
    return half or 1.0


def _render_one(
    pv,
    mesh_path: Path,
    out_path: Path,
    resolution: int,
    color: str,
    background: str,
    parallel_scale: float | None,
) -> None:
    mesh = pv.read(mesh_path)
    plotter = pv.Plotter(
        off_screen=True, shape=(2, 2), window_size=(resolution, resolution)
    )
    plotter.set_background(background)
    center = mesh.center

    for i, (label, azimuth, elevation) in enumerate(VIEWS):
        plotter.subplot(i // 2, i % 2)
        plotter.add_mesh(mesh, color=color, smooth_shading=True, specular=0.3)
        plotter.add_text(label, font_size=8, color="black")
        plotter.view_xy()  # deterministic base orientation
        plotter.camera.focal_point = center
        plotter.camera.azimuth = azimuth
        plotter.camera.elevation = elevation
        if parallel_scale is not None:
            # Set last: orientation changes can reset the parallel scale.
            plotter.enable_parallel_projection()
            plotter.camera.parallel_scale = parallel_scale
        else:
            plotter.reset_camera()

    plotter.screenshot(str(out_path))
    plotter.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="dir of {object_id}.ply meshes")
    parser.add_argument("output_dir", type=Path, help="dir to write {object_id}.png")
    parser.add_argument("--pattern", default="*.ply", help="glob for mesh files")
    parser.add_argument(
        "--organelle-type", default=None, help="for the sidecar + rubric"
    )
    parser.add_argument(
        "--collection-id", default=None, help="defaults to input dir name"
    )
    parser.add_argument("--resolution", type=int, default=1024, help="PNG side length")
    parser.add_argument("--color", default="lightgray", help="mesh color")
    parser.add_argument(
        "--background", default="white", help="neutral background color"
    )
    parser.add_argument(
        "--scale-mode",
        choices=["global", "per-object"],
        default="global",
        help="global keeps absolute size comparable across objects (recommended)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="render only the first N"
    )
    parser.add_argument(
        "--xvfb", action="store_true", help="start a virtual framebuffer"
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

    if args.xvfb:
        if hasattr(pv, "start_xvfb"):
            pv.start_xvfb()
        else:
            print(
                "this pyvista has no start_xvfb; wrap the call in `xvfb-run` "
                "instead, or rely on OSMesa/EGL.",
                file=sys.stderr,
            )
    pv.OFF_SCREEN = True

    if not args.input_dir.is_dir():
        print(f"not a directory: {args.input_dir}", file=sys.stderr)
        return 2

    paths = _iter_meshes(args.input_dir, args.pattern)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        print(f"no meshes matched {args.pattern} in {args.input_dir}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    collection_id = args.collection_id or args.input_dir.resolve().name

    parallel_scale = None
    if args.scale_mode == "global":
        parallel_scale = _global_half_extent(paths, pv) * 1.1  # small margin

    objects = []
    for mesh_path in tqdm(paths, desc="rendering", unit="obj"):
        object_id = mesh_path.stem
        out_path = args.output_dir / f"{object_id}.png"
        _render_one(
            pv,
            mesh_path,
            out_path,
            resolution=args.resolution,
            color=args.color,
            background=args.background,
            parallel_scale=parallel_scale,
        )
        objects.append({"object_id": object_id, "image": out_path.name, "meta": {}})

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
