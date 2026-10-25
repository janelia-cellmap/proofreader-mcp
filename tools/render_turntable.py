#!/usr/bin/env python
"""Render rotating viewpoints of one object's mesh for macro proofreading.

Phase 1 of the mesh-driven visual proofreading roadmap. Loads a single target
mesh, spins it about the screen-vertical axis under a fixed orthographic camera,
and composites a *stationary* labeled grid overlay (columns A.., rows 1..). Two
output modes:

* **video** (default) — a rotating MP4 with a per-frame HUD (frame/timestamp/angle).
* **stills** (``--stills N``) — N high-res labeled viewpoint PNGs. Recommended for
  VLMs: the model sees every view at full resolution, so it localizes far better
  than on a sub-sampled, downscaled video.

Either way it writes ``<id>.turntable.json`` — a sidecar recording the exact
camera, grid, rotation, and voxel scale so a (view/frame, grid cell or point) hit
can be inverted to a voxel coordinate (see grid_to_voxel.py).

Input sources mirror render_meshes.py (auto-detected, or forced with --source):

* **files** — a directory of ``{object_id}.ply`` (or .obj/.stl) meshes.
* **precomputed** — a Neuroglancer precomputed mesh layer (local or gs:///https).

Install deps and run (headless needs an offscreen GL context — wrap in xvfb-run)::

    uv sync --extra render --extra video --extra precomputed

    # 8 labeled stills (recommended for agents):
    xvfb-run -a uv run python tools/render_turntable.py \
        /path/to/mesh/multires 12345 /path/to/out \
        --source precomputed --voxel-size 8 8 8 --stills 8

    # or a turntable video:
    xvfb-run -a uv run python tools/render_turntable.py \
        /path/to/mesh/multires 12345 /path/to/out \
        --source precomputed --voxel-size 8 8 8

Output: ``<id>.view*.png`` (or ``<id>.mp4``) + ``<id>.turntable.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mesh_sources import make_source
from turntable_geometry import TurntableGeometry, column_label


def _box_half(points: np.ndarray, center: np.ndarray, up_axis: np.ndarray) -> float:
    """Half-size of a cube that contains the mesh at *every* spin angle.

    Spinning about ``up_axis`` sweeps each vertex around a circle of radius =
    its distance from the axis, so the in-plane extent must use that radius (not
    the static bounding box). Along the axis the extent is fixed. We take the max
    of both and pad slightly so the mesh never touches the wireframe.
    """
    rel = points - center
    axial = np.abs(rel @ _unit(up_axis))
    plane = np.linalg.norm(rel - np.outer(rel @ _unit(up_axis), _unit(up_axis)), axis=1)
    return float(max(plane.max(), axial.max())) * 1.05


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _load_font(size: int):
    """A bold TrueType font at ``size`` px, falling back to PIL's default."""
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_banner(frame_img, text: str, font) -> None:
    """Burn a top-left text banner over a dark strip into ``frame_img``."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(frame_img)
    box = draw.textbbox((0, 0), text, font=font)
    pad = 4
    draw.rectangle(
        [0, 0, box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad],
        fill=(0, 0, 0, 180),
    )
    draw.text((pad, pad), text, fill=(0, 255, 0, 255), font=font)


def _frame_hud_text(geom: TurntableGeometry, frame: int) -> str:
    """Per-frame video HUD: frame index, ``mm:ss.ss`` timestamp, spin angle."""
    seconds = frame / geom.fps
    ts = f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"
    az = np.rad2deg(geom.frame_angle_rad(frame))
    return f"frame {frame:03d}/{geom.n_frames:03d}   t={ts}   az={az:03.0f}°"


def _grid_overlay(geom: TurntableGeometry, line_rgba, label_rgba):
    """A stationary RGBA grid+labels overlay, drawn once and pasted on each frame."""
    from PIL import Image, ImageDraw

    w, h = geom.window
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = _load_font(max(12, h // 40))

    # Cell boundaries match turntable_geometry exactly: column i spans
    # [i/n_cols, (i+1)/n_cols] of the width; row j the analogous slice of height.
    for c in range(geom.n_cols + 1):
        x = round(c * w / geom.n_cols)
        draw.line([(x, 0), (x, h)], fill=line_rgba, width=1)
    for r in range(geom.n_rows + 1):
        y = round(r * h / geom.n_rows)
        draw.line([(0, y), (w, y)], fill=line_rgba, width=1)

    # One label per cell, anchored top-left inside the cell, so the agent can read
    # off "D4" directly. Column letter + row number, e.g. "D4".
    for c in range(geom.n_cols):
        for r in range(geom.n_rows):
            x = round(c * w / geom.n_cols) + 3
            y = round(r * h / geom.n_rows) + 2
            draw.text((x, y), f"{column_label(c)}{r + 1}", fill=label_rgba, font=font)
    return overlay


def _build_geometry(
    mesh,
    *,
    voxel_size_nm,
    forward,
    up_axis,
    origin_nm,
    n_cols,
    n_rows,
    fps,
    n_frames,
    total_degrees,
    window,
) -> TurntableGeometry:
    """Assemble the shared geometry record from a mesh + framing parameters."""
    center = np.asarray(mesh.center, dtype=float)
    points = np.asarray(mesh.points, dtype=float)
    box_half = _box_half(points, center, np.asarray(up_axis, float))
    return TurntableGeometry(
        center=center,
        forward=np.asarray(forward, float),
        up_axis=np.asarray(up_axis, float),
        # A small margin so the object never touches the frame edge.
        parallel_scale=box_half * 1.15,
        window=window,
        n_cols=n_cols,
        n_rows=n_rows,
        fps=fps,
        n_frames=n_frames,
        total_degrees=total_degrees,
        box_half=box_half,
        voxel_size_nm=np.asarray(np.broadcast_to(voxel_size_nm, (3,)), float),
        origin_nm=np.asarray(origin_nm, float),
    )


def _setup_plotter(geom: TurntableGeometry, window, background, pv):
    """A static orthographic plotter aimed at the object center.

    Under parallel projection the camera distance only affects clipping, not
    scale (fixed by ``parallel_scale``), so the framing is identical every frame.
    """
    up, fwd = geom.up, _unit(geom.forward)
    distance = geom.box_half * 4.0 + geom.parallel_scale
    plotter = pv.Plotter(off_screen=True, window_size=list(window))
    plotter.set_background(background)
    plotter.enable_parallel_projection()
    plotter.camera_position = [
        tuple(geom.center - fwd * distance),
        tuple(geom.center),
        tuple(up),
    ]
    plotter.camera.parallel_scale = geom.parallel_scale
    plotter.camera.clipping_range = (
        distance - geom.box_half * 2,
        distance + geom.box_half * 2,
    )
    return plotter


def _spin_screenshot(plotter, mesh, geom, frame, mesh_color, prev_actor):
    """Render one frame: spin the mesh to ``frame``'s angle, screenshot to RGBA."""
    from PIL import Image

    if prev_actor is not None:
        plotter.remove_actor(prev_actor)
    spun = mesh.copy()
    spun.rotate_vector(
        geom.up_axis,
        np.rad2deg(geom.frame_angle_rad(frame)),
        point=geom.center,
        inplace=True,
    )
    actor = plotter.add_mesh(spun, color=mesh_color, smooth_shading=True, specular=0.3)
    rgb = plotter.screenshot(return_img=True)
    return Image.fromarray(rgb).convert("RGBA"), actor


def render_turntable(
    mesh,
    out_mp4: Path,
    *,
    voxel_size_nm,
    forward=(0.0, -1.0, 0.0),
    up_axis=(0.0, 0.0, 1.0),
    origin_nm=(0.0, 0.0, 0.0),
    n_cols: int = 12,
    n_rows: int = 8,
    fps: int = 30,
    seconds: float = 3.0,
    total_degrees: float = 360.0,
    window=(960, 640),
    mesh_color: str = "lightgray",
    background: str = "black",
    overlays: bool = True,
) -> TurntableGeometry:
    """Render the turntable to ``out_mp4`` and return its geometry sidecar record.

    With ``overlays=False`` the frames carry nothing but the rotating mesh — no
    grid, labels, or HUD (the sidecar geometry is still written, so it can be
    mapped later).
    """
    import imageio.v2 as imageio
    import pyvista as pv

    pv.OFF_SCREEN = True

    geom = _build_geometry(
        mesh,
        voxel_size_nm=voxel_size_nm,
        forward=forward,
        up_axis=up_axis,
        origin_nm=origin_nm,
        n_cols=n_cols,
        n_rows=n_rows,
        fps=fps,
        n_frames=max(1, int(round(seconds * fps))),
        total_degrees=total_degrees,
        window=window,
    )
    plotter = _setup_plotter(geom, window, background, pv)
    overlay = None
    if overlays:
        overlay = _grid_overlay(
            geom, line_rgba=(255, 255, 255, 70), label_rgba=(255, 255, 0, 200)
        )
    hud_font = _load_font(max(14, window[1] // 32))

    mesh_actor = None
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_mp4), fps=fps, macro_block_size=8)
    try:
        for frame in range(geom.n_frames):
            frame_img, mesh_actor = _spin_screenshot(
                plotter, mesh, geom, frame, mesh_color, mesh_actor
            )
            if overlays:
                frame_img.alpha_composite(overlay)
                _draw_banner(frame_img, _frame_hud_text(geom, frame), hud_font)
            writer.append_data(np.asarray(frame_img.convert("RGB")))
    finally:
        writer.close()
        plotter.close()

    return geom


def render_stills(
    mesh,
    out_dir: Path,
    object_id: str,
    *,
    voxel_size_nm,
    n_views: int = 8,
    forward=(0.0, -1.0, 0.0),
    up_axis=(0.0, 0.0, 1.0),
    origin_nm=(0.0, 0.0, 0.0),
    n_cols: int = 6,
    n_rows: int = 6,
    total_degrees: float = 360.0,
    window=(900, 900),
    mesh_color: str = "lightgray",
    background: str = "black",
) -> tuple[TurntableGeometry, list[dict]]:
    """Render ``n_views`` labeled high-res still viewpoints + return their records.

    Stills sidestep the two things that break video for VLMs: the API never
    sub-samples them (the model sees *every* view) and they keep full resolution
    (the coarse grid + big view label stay readable). A "view" is just a frame at
    azimuth ``k / n_views * total_degrees``, so the same grid->voxel inversion
    applies (``grid_to_voxel.py`` treats ``view`` as the frame index).
    """
    import pyvista as pv

    pv.OFF_SCREEN = True

    geom = _build_geometry(
        mesh,
        voxel_size_nm=voxel_size_nm,
        forward=forward,
        up_axis=up_axis,
        origin_nm=origin_nm,
        n_cols=n_cols,
        n_rows=n_rows,
        fps=max(1, n_views),
        n_frames=n_views,
        total_degrees=total_degrees,
        window=window,
    )
    plotter = _setup_plotter(geom, window, background, pv)
    overlay = _grid_overlay(
        geom, line_rgba=(255, 255, 255, 90), label_rgba=(255, 255, 0, 220)
    )
    label_font = _load_font(max(20, window[1] // 22))

    out_dir.mkdir(parents=True, exist_ok=True)
    views: list[dict] = []
    actor = None
    try:
        for k in range(n_views):
            img, actor = _spin_screenshot(plotter, mesh, geom, k, mesh_color, actor)
            img.alpha_composite(overlay)
            az = float(np.rad2deg(geom.frame_angle_rad(k)))
            _draw_banner(img, f"VIEW {k}/{n_views}   az={az:03.0f}°", label_font)
            name = f"{object_id}.view{k}.png"
            img.convert("RGB").save(out_dir / name)
            views.append({"view": k, "angle_degrees": round(az, 1), "image": name})
    finally:
        plotter.close()

    return geom, views


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="mesh dir, or precomputed layer path/URL")
    parser.add_argument("object_id", help="id of the single object to render")
    parser.add_argument("output_dir", type=Path, help="dir for <id>.mp4 + sidecar")
    parser.add_argument(
        "--source", choices=["auto", "files", "precomputed"], default="auto"
    )
    parser.add_argument("--pattern", default="*.ply", help="glob for file source")
    parser.add_argument("--lod", type=int, default=0, help="precomputed LOD (0=finest)")
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs="+",
        default=[1.0, 1.0, 1.0],
        help="nm per voxel (one value, or three x y z) for grid->voxel mapping",
    )
    parser.add_argument(
        "--origin-nm",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="world nm at voxel (0,0,0); usually 0 0 0",
    )
    parser.add_argument(
        "--stills",
        type=int,
        default=None,
        metavar="N",
        help="emit N labeled still viewpoints (recommended for VLMs) instead of "
        "a video; the model localizes far better on full-res stills",
    )
    parser.add_argument(
        "--cols", type=int, default=None, help="grid columns (def 12 video / 6 stills)"
    )
    parser.add_argument(
        "--rows", type=int, default=None, help="grid rows (def 8 video / 6 stills)"
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=3.0, help="video length")
    parser.add_argument("--degrees", type=float, default=360.0, help="total rotation")
    parser.add_argument("--width", type=int, default=None, help="default 960/900")
    parser.add_argument("--height", type=int, default=None, help="default 640/900")
    parser.add_argument("--color", default="lightgray", help="mesh color")
    parser.add_argument("--background", default="black")
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="video only: render the bare rotating mesh (no grid/labels/HUD)",
    )
    args = parser.parse_args(argv)

    try:
        import imageio  # noqa: F401
        import pyvista  # noqa: F401
    except ImportError:
        print(
            "missing deps. Run: uv sync --extra render --extra video", file=sys.stderr
        )
        return 2

    kind = args.source
    source = make_source(args.input, kind=kind, pattern=args.pattern, lod=args.lod)
    try:
        mesh = source.get(args.object_id)
    finally:
        source.close()

    voxel_size = args.voxel_size if len(args.voxel_size) == 3 else args.voxel_size * 3
    sidecar = {
        "object_id": args.object_id,
        "source": {"input": args.input, "kind": kind, "lod": args.lod},
    }

    if args.stills:
        cols = args.cols if args.cols is not None else 6
        rows = args.rows if args.rows is not None else 6
        window = (args.width or 900, args.height or 900)
        geom, views = render_stills(
            mesh,
            args.output_dir,
            args.object_id,
            voxel_size_nm=voxel_size,
            n_views=args.stills,
            origin_nm=args.origin_nm,
            n_cols=cols,
            n_rows=rows,
            total_degrees=args.degrees,
            window=window,
            mesh_color=args.color,
            background=args.background,
        )
        sidecar.update({"mode": "stills", "views": views, "geometry": geom.to_dict()})
        sidecar_path = args.output_dir / f"{args.object_id}.turntable.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2))
        stem = f"{args.output_dir}/{args.object_id}.view*.png"
        print(f"wrote {len(views)} stills -> {stem}")
        print(f"wrote {sidecar_path}")
        return 0

    cols = args.cols if args.cols is not None else 12
    rows = args.rows if args.rows is not None else 8
    window = (args.width or 960, args.height or 640)
    out_mp4 = args.output_dir / f"{args.object_id}.mp4"
    geom = render_turntable(
        mesh,
        out_mp4,
        voxel_size_nm=voxel_size,
        origin_nm=args.origin_nm,
        n_cols=cols,
        n_rows=rows,
        fps=args.fps,
        seconds=args.seconds,
        total_degrees=args.degrees,
        window=window,
        mesh_color=args.color,
        background=args.background,
        overlays=not args.no_overlay,
    )
    sidecar.update({"mode": "video", "video": out_mp4.name, "geometry": geom.to_dict()})
    sidecar_path = args.output_dir / f"{args.object_id}.turntable.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    print(f"wrote {out_mp4} ({geom.n_frames} frames @ {geom.fps}fps)")
    print(f"wrote {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
