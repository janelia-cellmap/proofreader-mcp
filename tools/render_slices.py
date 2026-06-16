#!/usr/bin/env python
"""Render raw-EM + segmentation 2D cross-section montages for proofreading.

Second-pass companion to ``render_meshes.py``. For each object it reads the raw
EM and the voxel segmentation (OME-Zarr) around the object's mesh-derived
centroid and renders the three orthogonal center sections (perpendicular to X, Y,
Z), each with the mesh viewed straight down the same axis above it so they line up:

    [ mesh down X ] [ mesh down Y ] [ mesh down Z ]
    [  X section  ] [  Y section  ] [  Z section  ]

Each column is one axis (border-colored X red / Y green / Z blue). The crop is a
cube (same nm field of view in every axis), so all panels -- sections and mesh
views -- share ONE nm/pixel scale; an object therefore looks smaller in the view
down its long axis and larger in the others, instead of being stretched to fill.
Sections are raw EM (grayscale, fixed contrast window) with the object's
segmentation overlaid as a light translucent tint.

Intended to run STAGED: classify meshes first, then render slices only for the
flagged subset (``--labels likely_overmerged,likely_oversegmented,uncertain``).

Usage::

    uv sync --extra slices --extra render --extra precomputed
    uv run python tools/render_slices.py ignore/snapshots_lod0_100 OUT_DIR \\
        --raw /nrs/cellmap/data/.../em/fibsem-uint8 \\
        --seg /nrs/cellmap/.../jrc_P3_E5_D1_N2.zarr/mito \\
        --mesh-source /nrs/cellmap/.../mito/multires \\
        --contrast 84,152 \\
        --labels likely_overmerged,likely_oversegmented,uncertain
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont

from proofreader_mcp.storage import Store

# (label, axis index in z,y,x order, RGB color).
PLANES = [
    ("X", 2, (220, 60, 60)),
    ("Y", 1, (40, 175, 70)),
    ("Z", 0, (70, 120, 225)),
]
# Mesh camera per section axis (z,y,x index): (view direction, up) in x,y,z. Chosen
# so the mesh's screen axes match the (flipud'd) section: X->+X, Y->-Y, Z->+Z, all
# with the section's horizontal/vertical world axes lined up.
MESH_VIEW = {
    2: ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # X section: view down +X, up +Z
    1: ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),  # Y section: view down -Y, up +Z
    0: ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),  # Z section: view down +Z, up +Y
}
OVERLAY_RGB = (235, 90, 90)
OVERLAY_ALPHA = 0.10  # very light tint so the EM underneath stays clearly readable
MESH_OPACITY = 1.0
SEP = 6
SEP_COLOR = (30, 30, 30)


def open_level0(group_path: Path):
    """Open an OME-Zarr group's level-0 array; return (array, scale, translation).

    scale/translation are in z,y,x order, in nm.
    """
    attrs = json.loads((group_path / ".zattrs").read_text())
    ds0 = attrs["multiscales"][0]["datasets"][0]
    scale, translation = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]
    for ct in ds0["coordinateTransformations"]:
        if ct["type"] == "scale":
            scale = ct["scale"]
        elif ct["type"] == "translation":
            translation = ct["translation"]
    arr = zarr.open(str(group_path / ds0["path"]), mode="r")
    return arr, scale, translation


def _crop_box(center_zyx, half_zyx, scale, translation, shape):
    """nm box -> clamped voxel index ranges per axis."""
    out = []
    for a in range(3):
        lo = (center_zyx[a] - half_zyx[a] - translation[a]) / scale[a]
        hi = (center_zyx[a] + half_zyx[a] - translation[a]) / scale[a]
        out.append((max(0, int(np.floor(lo))), min(shape[a], int(np.ceil(hi)) + 1)))
    return out


def _gray_rgb(slice2d, lo, hi):
    """Fixed-window contrast -> grayscale RGB."""
    norm = np.clip((slice2d.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    g = (norm * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def _overlay(rgb, mask2d):
    if mask2d.any():
        tint = np.array(OVERLAY_RGB, dtype=np.float32)
        blended = rgb.astype(np.float32) * (1 - OVERLAY_ALPHA) + tint * OVERLAY_ALPHA
        rgb = np.where(mask2d[..., None], blended.astype(np.uint8), rgb)
    return rgb


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _bordered(img: Image.Image, color, width: int) -> Image.Image:
    d = ImageDraw.Draw(img)
    for w in range(width):
        d.rectangle([w, w, img.width - 1 - w, img.height - 1 - w], outline=color)
    return img


def _section_panel(raw_crop, seg_crop, axis, obj_id, cell, lo, hi):
    """Center section perpendicular to `axis` (z,y,x index): raw + light overlay.

    flipud puts the higher world coordinate at the top so the section's orientation
    matches the axis-aligned mesh view above it.
    """
    raw2d = np.flipud(np.take(raw_crop, raw_crop.shape[axis] // 2, axis=axis))
    seg2d = np.flipud(np.take(seg_crop, seg_crop.shape[axis] // 2, axis=axis)) == int(
        obj_id
    )
    base = Image.fromarray(_gray_rgb(raw2d, lo, hi)).resize((cell, cell), Image.NEAREST)
    mask = Image.fromarray((seg2d * 255).astype(np.uint8)).resize(
        (cell, cell), Image.NEAREST
    )
    return Image.fromarray(_overlay(np.asarray(base), np.asarray(mask) > 127))


def _mesh_view(pv, mesh, axis_zyx, center_xyz, extent_xyz, cell, parallel_scale):
    """Solid mesh viewed straight down `axis`, oriented to line up with its section.

    ``parallel_scale`` is shared across all three axes (= half the cubic field of
    view) so the mesh views and the sections all share one nm/pixel scale.
    """
    direction, up = MESH_VIEW[axis_zyx]
    pl = pv.Plotter(off_screen=True, window_size=(cell, cell))
    pl.set_background("white")
    # Match the mesh-montage look: lightgray, smooth shading, a little specular.
    pl.add_mesh(
        mesh,
        color="lightgray",
        opacity=MESH_OPACITY,
        smooth_shading=True,
        specular=0.3,
    )
    dist = max(extent_xyz) * 4
    pos = [center_xyz[j] + direction[j] * dist for j in range(3)]
    pl.camera_position = [tuple(pos), tuple(center_xyz), up]
    pl.enable_parallel_projection()
    pl.camera.parallel_scale = parallel_scale
    arr = pl.screenshot(return_img=True)
    pl.close()
    return Image.fromarray(arr).resize((cell, cell))


def render_object(
    obj_id,
    center_xyz,
    extent_xyz,
    raw,
    raw_st,
    seg,
    seg_st,
    mesh,
    pv,
    cell,
    margin,
    lo,
    hi,
):
    rscale, rtrans = raw_st
    sscale, strans = seg_st
    center_zyx = [center_xyz[2], center_xyz[1], center_xyz[0]]
    # Cubic field of view (same nm box in z/y/x) so every section -- and the mesh
    # views -- share one nm/pixel scale, regardless of the object's aspect ratio.
    fov_half = max(extent_xyz) / 2 * margin
    half_zyx = [fov_half, fov_half, fov_half]

    rb = _crop_box(center_zyx, half_zyx, rscale, rtrans, raw.shape)
    sb = _crop_box(center_zyx, half_zyx, sscale, strans, seg.shape)
    raw_crop = np.asarray(
        raw[rb[0][0] : rb[0][1], rb[1][0] : rb[1][1], rb[2][0] : rb[2][1]]
    )
    seg_crop = np.asarray(
        seg[sb[0][0] : sb[0][1], sb[1][0] : sb[1][1], sb[2][0] : sb[2][1]]
    )
    if raw_crop.size == 0:
        return None

    font = _font(max(14, cell // 22))
    have_mesh = mesh is not None
    rows = 2 if have_mesh else 1
    border = max(3, cell // 90)
    canvas = Image.new(
        "RGB",
        (3 * cell + 4 * SEP, rows * cell + (rows + 1) * SEP),
        SEP_COLOR,
    )

    for col, (label, axis, color) in enumerate(PLANES):
        x = SEP + col * (cell + SEP)
        if have_mesh:
            mv = _bordered(
                _mesh_view(pv, mesh, axis, center_xyz, extent_xyz, cell, fov_half),
                color,
                border,
            )
            ImageDraw.Draw(mv).text((8, 6), label, fill=color, font=font)
            canvas.paste(mv, (x, SEP))
        sec = _bordered(
            _section_panel(raw_crop, seg_crop, axis, obj_id, cell, lo, hi),
            color,
            border,
        )
        if not have_mesh:
            ImageDraw.Draw(sec).text((8, 6), label, fill=color, font=font)
        canvas.paste(sec, (x, SEP + (rows - 1) * (cell + SEP)))
    return canvas


def select_ids(store, cid, ids_arg, labels_arg, limit):
    col = store.get(cid)
    if ids_arg:
        ids = [s.strip() for s in ids_arg.split(",") if s.strip()]
    elif labels_arg:
        want = {s.strip() for s in labels_arg.split(",")}
        ranked = sorted(
            (v for v in col.verdicts.values() if v.primary_label.value in want),
            key=lambda v: -v.confidence,
        )
        ids = [v.object_id for v in ranked]
    else:
        ids = sorted(col.objects, key=int)
    return ids[:limit] if limit else ids


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("collection_dir", type=Path, help="dir with objects.json (+ state)")
    p.add_argument("output_dir", type=Path, help="dir to write {id}.png montages")
    p.add_argument("--raw", required=True, type=Path, help="OME-Zarr raw EM group")
    p.add_argument(
        "--seg", required=True, type=Path, help="OME-Zarr segmentation group"
    )
    p.add_argument("--mesh-source", default=None, help="mesh source for the top row")
    p.add_argument(
        "--mesh-lod", type=int, default=0, help="LOD for the mesh views (0=finest)"
    )
    p.add_argument(
        "--ids", default=None, help="comma-separated ids (overrides --labels)"
    )
    p.add_argument(
        "--labels",
        default=None,
        help="render only objects whose verdict primary_label is in this list "
        "(comma-separated), highest-confidence first",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--cell-size", type=int, default=420, help="pixels per panel")
    p.add_argument("--margin", type=float, default=1.5, help="crop padding factor")
    p.add_argument(
        "--contrast",
        default="84,152",
        help="fixed raw-EM intensity window lo,hi (uint8) applied to every slice",
    )
    args = p.parse_args(argv)

    lo, hi = (float(x) for x in args.contrast.split(","))

    store = Store()
    rep = store.load_collection(args.collection_dir)
    cid = rep.collection_id
    col = store.get(cid)
    ids = select_ids(store, cid, args.ids, args.labels, args.limit)
    if not ids:
        print("no objects selected (check --ids/--labels and prior classification)")
        return 1

    raw, rscale, rtrans = open_level0(args.raw)
    seg, sscale, strans = open_level0(args.seg)
    print(
        f"raw {raw.shape}@{rscale}nm  seg {seg.shape}@{sscale}nm  ->  {len(ids)} objects"
    )

    pv = mesh_source = None
    if args.mesh_source:
        import pyvista as pv

        from mesh_sources import make_source

        pv.OFF_SCREEN = True
        mesh_source = make_source(args.mesh_source, kind="auto", lod=args.mesh_lod)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    try:
        for oid in ids:
            meta = col.objects[oid].meta or {}
            if "centroid_nm" not in meta or "extent_nm" not in meta:
                print(f"  skip {oid}: no centroid/extent in objects.json")
                continue
            mesh = None
            if mesh_source is not None:
                try:
                    mesh = mesh_source.get(oid)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {oid}: mesh unavailable ({exc}); slices only")
            grid = render_object(
                oid,
                meta["centroid_nm"],
                meta["extent_nm"],
                raw,
                (rscale, rtrans),
                seg,
                (sscale, strans),
                mesh,
                pv,
                args.cell_size,
                args.margin,
                lo,
                hi,
            )
            if grid is None:
                print(f"  skip {oid}: empty crop")
                continue
            grid.save(args.output_dir / f"{oid}.png")
            done += 1
    finally:
        if mesh_source is not None:
            mesh_source.close()
    print(f"wrote {done} montages -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
