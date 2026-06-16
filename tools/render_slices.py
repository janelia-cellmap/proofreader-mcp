#!/usr/bin/env python
"""Render raw-EM + segmentation 2D cross-section montages for proofreading.

Second-pass companion to ``render_meshes.py``. For each object it reads the raw
EM and the voxel segmentation (OME-Zarr) around the object's mesh-derived
centroid and renders the three orthogonal center sections (perpendicular to X, Y,
Z) through the object, so a reviewer can confirm a mesh-based call against the
actual data (e.g. is an apparent overmerge a real thin neck?).

Each output PNG is a row of panels separated by dividers:

    [ X section ] [ Y section ] [ Z section ] [ mesh + cut planes ]

The three section panels show raw EM (grayscale, per-slice auto-contrast) with the
object's own segmentation overlaid in a light translucent tint, and a colored
border per axis (X red, Y green, Z blue). The optional 4th panel (when
``--mesh-source`` is given) shows the object's mesh with the three cutting planes
drawn in the matching colors, so you can see where each section was taken.

Intended to run STAGED: classify meshes first, then render slices only for the
flagged subset (``--labels likely_overmerged,likely_oversegmented,uncertain``).

Usage::

    uv sync --extra slices --extra render --extra precomputed
    uv run python tools/render_slices.py ignore/snapshots_lod0_100 OUT_DIR \\
        --raw /nrs/cellmap/data/.../em/fibsem-uint8 \\
        --seg /nrs/cellmap/.../jrc_P3_E5_D1_N2.zarr/mito \\
        --mesh-source /nrs/cellmap/.../mito/multires \\
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

# (label, axis index in z,y,x order, RGB color). Color ties each section panel to
# its plane in the mesh panel.
PLANES = [
    ("X", 2, (220, 60, 60)),
    ("Y", 1, (40, 175, 70)),
    ("Z", 0, (70, 120, 225)),
]
OVERLAY_RGB = (235, 90, 90)
OVERLAY_ALPHA = 0.20  # light tint so the EM underneath stays readable
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


def _auto_gray_rgb(slice2d, plo, phi):
    """Per-slice percentile auto-contrast -> grayscale RGB."""
    lo, hi = np.percentile(slice2d, [plo, phi])
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


def _section_panel(raw_crop, seg_crop, axis, obj_id, color, cell, plo, phi, rb, sb):
    """Center section perpendicular to `axis`: raw + light seg overlay, bordered."""
    ri = raw_crop.shape[axis] // 2
    si = seg_crop.shape[axis] // 2
    raw2d = np.take(raw_crop, ri, axis=axis)
    seg2d = np.take(seg_crop, si, axis=axis) == int(obj_id)
    base = Image.fromarray(_auto_gray_rgb(raw2d, plo, phi)).resize(
        (cell, cell), Image.NEAREST
    )
    mask = Image.fromarray((seg2d * 255).astype(np.uint8)).resize(
        (cell, cell), Image.NEAREST
    )
    comp = Image.fromarray(_overlay(np.asarray(base), np.asarray(mask) > 127))
    return _bordered(comp, color, max(3, cell // 90))


def _mesh_panel(pv, mesh, center, extents, cell):
    """Mesh with the three orthogonal cut planes drawn in the axis colors."""
    pl = pv.Plotter(off_screen=True, window_size=(cell, cell))
    pl.set_background("white")
    pl.add_mesh(mesh, color="lightgray", opacity=0.55, smooth_shading=True)
    size = max(extents) * 1.25
    for _label, axis, color in PLANES:
        normal = [0.0, 0.0, 0.0]
        normal[2 - axis] = 1.0  # axis is z,y,x index; convert to x,y,z normal
        plane = pv.Plane(center=center, direction=normal, i_size=size, j_size=size)
        pl.add_mesh(
            plane, color=[c / 255 for c in color], opacity=0.32, show_edges=False
        )
    pl.camera_position = "iso"
    pl.camera.zoom(1.2)
    arr = pl.screenshot(return_img=True)
    pl.close()
    return _bordered(Image.fromarray(arr).resize((cell, cell)), SEP_COLOR, 2)


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
    plo,
    phi,
):
    rscale, rtrans = raw_st
    sscale, strans = seg_st
    center_zyx = [center_xyz[2], center_xyz[1], center_xyz[0]]
    extent_zyx = [extent_xyz[2], extent_xyz[1], extent_xyz[0]]
    half_zyx = [e / 2 * margin for e in extent_zyx]

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
    panels = []
    for label, axis, color in PLANES:
        panel = _section_panel(
            raw_crop, seg_crop, axis, obj_id, color, cell, plo, phi, rb, sb
        )
        ImageDraw.Draw(panel).text((8, 6), label, fill=color, font=font)
        panels.append(panel)
    if mesh is not None:
        mp = _mesh_panel(pv, mesh, center_xyz, extent_xyz, cell)
        ImageDraw.Draw(mp).text((8, 6), "mesh", fill=(40, 40, 40), font=font)
        panels.append(mp)

    n = len(panels)
    canvas = Image.new("RGB", (n * cell + (n + 1) * SEP, cell + 2 * SEP), SEP_COLOR)
    for i, panel in enumerate(panels):
        canvas.paste(panel, (SEP + i * (cell + SEP), SEP))
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
    p.add_argument("--mesh-source", default=None, help="mesh source for the 4th panel")
    p.add_argument("--mesh-lod", type=int, default=2, help="LOD for the mesh panel")
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
    p.add_argument("--contrast", default="2,98", help="auto-contrast percentiles lo,hi")
    args = p.parse_args(argv)

    plo, phi = (float(x) for x in args.contrast.split(","))

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
                plo,
                phi,
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
