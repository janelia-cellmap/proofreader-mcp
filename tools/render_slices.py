#!/usr/bin/env python
"""Render raw-EM + segmentation 2D cross-section montages for proofreading.

Second-pass companion to ``render_meshes.py``. For each object it reads the raw
EM and the voxel segmentation (OME-Zarr) around the object's mesh-derived
centroid and renders orthogonal cross-sections, so a reviewer can confirm a
mesh-based call against the actual data (e.g. is an apparent overmerge a real
thin neck, is an oversegment a true membrane boundary).

Each output PNG is a 3x3 montage: rows are the three section planes (perpendicular
to X, Y, Z -- matching the mesh montage's axis columns), columns are three slice
positions through the object (-1/4, center, +1/4 of the extent). Raw EM is shown
grayscale with per-crop percentile contrast; the object's own segmentation is
overlaid translucent red.

Intended to run STAGED: classify meshes first, then render slices only for the
flagged subset (``--labels likely_overmerged,likely_oversegmented,uncertain``).

Usage::

    uv sync --extra slices
    uv run python tools/render_slices.py ignore/snapshots_lod0_100 OUT_DIR \\
        --raw  /nrs/cellmap/data/.../em/fibsem-uint8 \\
        --seg  /nrs/cellmap/.../jrc_P3_E5_D1_N2.zarr/mito \\
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

# Section planes: (label, axis index in z,y,x order). Rows of the montage.
PLANES = [("X", 2), ("Y", 1), ("Z", 0)]
POS_FRACS = (-0.25, 0.0, 0.25)  # columns: offsets along the plane's axis
OVERLAY_RGB = (220, 40, 40)
OVERLAY_ALPHA = 0.38


def open_level0(group_path: Path):
    """Open an OME-Zarr group's level-0 array; return (array, scale, translation).

    scale/translation are in z,y,x order, in nm (per the dataset's
    coordinateTransformations).
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
    """nm box -> clamped voxel index ranges (z0,z1,y0,y1,x0,x1)."""
    out = []
    for a in range(3):
        lo = (center_zyx[a] - half_zyx[a] - translation[a]) / scale[a]
        hi = (center_zyx[a] + half_zyx[a] - translation[a]) / scale[a]
        out.append((max(0, int(np.floor(lo))), min(shape[a], int(np.ceil(hi)) + 1)))
    return out


def _to_gray_rgb(slice2d, lo, hi):
    norm = np.clip((slice2d.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    g = (norm * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def _overlay(rgb, mask2d):
    if mask2d.any():
        m = mask2d[..., None]
        tint = np.array(OVERLAY_RGB, dtype=np.float32)
        blended = rgb.astype(np.float32) * (1 - OVERLAY_ALPHA) + tint * OVERLAY_ALPHA
        rgb = np.where(m, blended.astype(np.uint8), rgb)
    return rgb


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_object_slices(
    obj_id, center_xyz, extent_xyz, raw, raw_st, seg, seg_st, cell, margin
):
    rscale, rtrans = raw_st
    sscale, strans = seg_st
    center_zyx = [center_xyz[2], center_xyz[1], center_xyz[0]]
    extent_zyx = [extent_xyz[2], extent_xyz[1], extent_xyz[0]]
    half_zyx = [e / 2 * margin for e in extent_zyx]

    rb = _crop_box(center_zyx, half_zyx, rscale, rtrans, raw.shape)
    sb = _crop_box(center_zyx, half_zyx, sscale, strans, seg.shape)
    raw_crop = np.asarray(raw[rb[0][0]:rb[0][1], rb[1][0]:rb[1][1], rb[2][0]:rb[2][1]])
    seg_crop = np.asarray(seg[sb[0][0]:sb[0][1], sb[1][0]:sb[1][1], sb[2][0]:sb[2][1]])
    if raw_crop.size == 0:
        return None
    lo, hi = np.percentile(raw_crop, [1, 99])

    grid = Image.new("RGB", (cell * 3, cell * 3), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    font = _font(max(13, cell // 22))

    for row, (plane, axis) in enumerate(PLANES):
        for col, frac in enumerate(POS_FRACS):
            pos_nm = center_zyx[axis] + frac * extent_zyx[axis]
            ri = int(round((pos_nm - rtrans[axis]) / rscale[axis])) - rb[axis][0]
            si = int(round((pos_nm - strans[axis]) / sscale[axis])) - sb[axis][0]
            ri = min(max(ri, 0), raw_crop.shape[axis] - 1)
            si = min(max(si, 0), seg_crop.shape[axis] - 1)
            raw2d = np.take(raw_crop, ri, axis=axis)
            seg2d = np.take(seg_crop, si, axis=axis) == int(obj_id)

            # Resize raw and mask to the panel independently (the two volumes have
            # different voxel sizes but cover the same nm box), then overlay.
            base = Image.fromarray(_to_gray_rgb(raw2d, lo, hi)).resize(
                (cell, cell), Image.NEAREST
            )
            mask = Image.fromarray((seg2d * 255).astype(np.uint8)).resize(
                (cell, cell), Image.NEAREST
            )
            comp = _overlay(np.asarray(base), np.asarray(mask) > 127)
            grid.paste(Image.fromarray(comp), (col * cell, row * cell))
            label = f"{plane} {'+' if frac >= 0 else ''}{int(frac * extent_zyx[axis])}nm"
            d = ImageDraw.Draw(grid)
            d.text((col * cell + 6, row * cell + 4), label, fill=(255, 70, 70), font=font)
    draw.text((6, cell * 3 - max(13, cell // 22) - 8),
              f"id {obj_id}  raw EM + seg overlay", fill=(0, 0, 0), font=font)
    return grid


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
    p.add_argument("output_dir", type=Path, help="dir to write {id}.png slice montages")
    p.add_argument("--raw", required=True, type=Path, help="OME-Zarr raw EM group")
    p.add_argument("--seg", required=True, type=Path, help="OME-Zarr segmentation group")
    p.add_argument("--ids", default=None, help="comma-separated ids (overrides --labels)")
    p.add_argument(
        "--labels",
        default=None,
        help="render only objects whose verdict primary_label is in this list "
        "(comma-separated), highest-confidence first; needs prior classification",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--cell-size", type=int, default=384, help="pixels per panel")
    p.add_argument("--margin", type=float, default=1.5, help="crop padding factor")
    args = p.parse_args(argv)

    store = Store()
    rep = store.load_collection(args.collection_dir)
    cid = rep.collection_id
    col = store.get(cid)
    ids = select_ids(store, cid, args.ids, args.labels, args.limit)
    if not ids:
        print("no objects selected (check --ids/--labels and that meshes were classified)")
        return 1

    raw, rscale, rtrans = open_level0(args.raw)
    seg, sscale, strans = open_level0(args.seg)
    print(f"raw {raw.shape}@{rscale}nm  seg {seg.shape}@{sscale}nm  ->  {len(ids)} objects")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for oid in ids:
        meta = col.objects[oid].meta or {}
        if "centroid_nm" not in meta or "extent_nm" not in meta:
            print(f"  skip {oid}: no centroid/extent in objects.json")
            continue
        grid = render_object_slices(
            oid, meta["centroid_nm"], meta["extent_nm"],
            raw, (rscale, rtrans), seg, (sscale, strans),
            args.cell_size, args.margin,
        )
        if grid is None:
            print(f"  skip {oid}: empty crop")
            continue
        grid.save(args.output_dir / f"{oid}.png")
        done += 1
    print(f"wrote {done} slice montages -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
