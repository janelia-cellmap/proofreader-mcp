#!/usr/bin/env python
"""Render raw-EM + segmentation 2D cross-section montages for proofreading.

Second-pass companion to ``render_meshes.py``. For each object it reads the raw
EM and the voxel segmentation (OME-Zarr) around the object's mesh-derived centroid
and renders, per axis ROW (X/Y/Z), a front + back mesh view plus N sections
stepping through the object along that axis (near -> far, left -> right):

    X | [+X front] [slice1 blue] ... [center] ... [sliceN red] [-X back]
    Y | [+Y front] [slice1 blue] ... [center] ... [sliceN red] [-Y back]
    Z | [+Z front] [slice1 blue] ... [center] ... [sliceN red] [-Z back]

The mesh is colored by position along the axis (blue near -> red far); the front
view shows the near/blue side and the back view (camera flipped 180 deg) shows the
far/red side, which the front can't. Every slice's border uses the same color map,
so a blue-bordered slice corresponds to the blue part of the mesh, etc.
The crop is a cube (same nm field of view in every axis), so all panels share ONE
nm/pixel scale (an object looks smaller down its long axis, larger in the others).
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
import colorsys
import json
from pathlib import Path

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from proofreader_mcp.storage import Store

# (label, axis index in z,y,x order, RGB color).
PLANES = [
    ("X", 2, (220, 60, 60)),
    ("Y", 1, (40, 175, 70)),
    ("Z", 0, (70, 120, 225)),
]
# Front mesh camera per section axis (z,y,x index): (view direction, up) in x,y,z.
# All fronts look down the +axis so the left ("front") panel is always +X/+Y/+Z.
MESH_VIEW = {
    2: ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # X: view down +X, up +Z
    1: ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),  # Y: view down +Y, up +Z
    0: ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),  # Z: view down +Z, up +Y
}
# Sections whose horizontal axis must be flipped to match the +axis mesh view.
SECTION_HFLIP = {1}  # Y: +Y view has screen-right = -X, so mirror the section
OVERLAY_RGB = (235, 90, 90)
OVERLAY_ALPHA = 0.10  # very light tint so the EM underneath stays clearly readable
MESH_OPACITY = 1.0
SEP = 2
SEP_COLOR = (30, 30, 30)
SPAN_FRAC = 0.45  # slices span centroid +/- SPAN_FRAC * extent along the axis
VIEW_PAD = 1.1  # --fit view: window = in-plane extent * this (tight fill, not margin)

# Description of the montage layout, written into the collection sidecar so an MCP
# host knows how to read each object image when classifying.
LAYOUT_LEGEND = (
    "Each object image is a 3-row montage, one row per axis (top=X, mid=Y, bot=Z). "
    "Reading each row left to right: (1) a 3D mesh of the object viewed down the "
    "+axis (front/near side); (2) N raw-EM cross-sections stepping through the "
    "object from near to far along that axis; (3) the same mesh from the -axis "
    "(back/far side, mirrored so it lines up with the sections). The mesh surface is "
    "colored by depth along the axis (blue=near -> red=far) and each section's "
    "colored border matches where it sits in that gradient. In each section the raw "
    "EM is grayscale (fixed contrast), the object's own voxels are faintly tinted, a "
    "thin ring of surrounding EM (halo) gives context, and everything beyond is "
    "white. Each mesh panel is labeled with the object's extent along that axis in "
    "nm. Use the mesh for overall shape (overmerge/oversegment) and the EM sections "
    "to judge whether the object is a real, correctly-classified organelle."
)


def _depth_color(t: float) -> tuple[int, int, int]:
    """Map normalized depth t in [0, 1] to an RGB color (blue near -> red far)."""
    t = min(max(t, 0.0), 1.0)
    r, g, b = colorsys.hsv_to_rgb(0.66 * (1.0 - t), 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


# 256-entry lookup table for fast per-vertex depth coloring of the mesh.
_DEPTH_LUT = np.array([_depth_color(i / 255) for i in range(256)], dtype=np.uint8)


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


def _overlay(rgb, mask2d, tint_rgb=OVERLAY_RGB):
    if mask2d.any():
        tint = np.array(tint_rgb, dtype=np.float32)
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


def _center_crop(arr, frac):
    """Keep the central ``frac`` of a 2D array (centered on the object centroid)."""
    if frac >= 1.0:
        return arr
    h, w = arr.shape[:2]
    ch, cw = max(1, int(round(h * frac))), max(1, int(round(w * frac)))
    r0, c0 = (h - ch) // 2, (w - cw) // 2
    return arr[r0 : r0 + ch, c0 : c0 + cw]


def _bbox_crop(raw2d, seg2d):
    """Crop both arrays to this slice's object bounding box (square, small pad).

    Used by --fit panel so each section is zoomed to its own cross-section. The box
    is computed in fractions of the (square, cubic) read window, so the same box
    applies to raw and seg despite their different voxel sizes. The pad is small and
    fixed; the mask halo (--mask-pad) fills it with surrounding EM rather than white.
    """
    ys, xs = np.where(seg2d)
    if ys.size == 0:
        return raw2d, seg2d
    h, w = seg2d.shape
    padf = 0.05
    t = max(0.0, ys.min() / h - padf)
    b = min(1.0, (ys.max() + 1) / h + padf)
    left = max(0.0, xs.min() / w - padf)
    right = min(1.0, (xs.max() + 1) / w + padf)
    side = min(1.0, max(b - t, right - left))
    cy, cx = (t + b) / 2, (left + right) / 2
    t = min(max(0.0, cy - side / 2), 1.0 - side)
    left = min(max(0.0, cx - side / 2), 1.0 - side)
    b, right = t + side, left + side

    def crop(a):
        ah, aw = a.shape[:2]
        r0, r1 = int(t * ah), max(int(t * ah) + 1, int(b * ah))
        c0, c1 = int(left * aw), max(int(left * aw) + 1, int(right * aw))
        return a[r0:r1, c0:c1]

    return crop(raw2d), crop(seg2d)


def _section_at(
    raw_crop,
    seg_crop,
    axis,
    ri,
    si,
    obj_id,
    cell,
    lo,
    hi,
    tint,
    mask_mode,
    pad_vox,
    fit,
    fit_frac,
):
    """Section perpendicular to `axis` (z,y,x index) at raw index ri / seg index si.

    flipud puts the higher world coordinate at the top so the section's orientation
    matches the axis-aligned mesh view above it. ``mask_mode`` controls how the
    object is shown: 'off' = translucent ``tint`` overlay; 'object' = keep raw only
    inside the segmentation (white outside); 'dim' = darken everything outside.
    ``pad_vox`` dilates the kept region to leave a ring of surrounding EM context.
    ``fit``: 'cube' no crop; 'view' center-crop to ``fit_frac`` (per-row scale);
    'panel' crop to this slice's own object bbox (per-slice zoom).
    """
    ri = min(max(ri, 0), raw_crop.shape[axis] - 1)
    si = min(max(si, 0), seg_crop.shape[axis] - 1)
    raw2d = np.flipud(np.take(raw_crop, ri, axis=axis))
    seg2d = np.flipud(np.take(seg_crop, si, axis=axis)) == int(obj_id)
    if fit == "view":
        raw2d, seg2d = _center_crop(raw2d, fit_frac), _center_crop(seg2d, fit_frac)
    elif fit == "panel":
        raw2d, seg2d = _bbox_crop(raw2d, seg2d)
    if axis in SECTION_HFLIP:
        raw2d = np.fliplr(raw2d)
        seg2d = np.fliplr(seg2d)
    base = np.asarray(
        Image.fromarray(_gray_rgb(raw2d, lo, hi)).resize((cell, cell), Image.NEAREST)
    )
    seg_native = Image.fromarray((seg2d * 255).astype(np.uint8))
    obj_mask = np.asarray(seg_native.resize((cell, cell), Image.NEAREST)) > 127
    halo = seg_native
    if pad_vox > 0:
        # Dilate at the segmentation's native resolution: the kernel is a constant
        # ~pad_vox voxels regardless of object size / panel size (cheap).
        halo = seg_native.filter(ImageFilter.MaxFilter(2 * pad_vox + 1))
    halo_mask = np.asarray(halo.resize((cell, cell), Image.NEAREST)) > 127
    if mask_mode == "object":
        out = np.full_like(base, 255)
        out[halo_mask] = base[halo_mask]
    elif mask_mode == "dim":
        out = (base.astype(np.float32) * 0.3).astype(np.uint8)
        out[halo_mask] = base[halo_mask]
    else:  # "off": full EM
        out = base.copy()
    # Faint tint on the TRUE object voxels so it reads against the EM context/halo.
    out = _overlay(out, obj_mask, tint)
    return Image.fromarray(out)


def _depth_colors(mesh, ax, s, center_xyz, extent_xyz):
    """Per-vertex RGB encoding distance from the FRONT camera along axis `ax`.

    blue = nearest the front camera, red = farthest. ``s`` is the front camera's
    side (+1/-1). The same colors are reused for the back view, so the back shows
    the red (far) side -- which the front view can't.
    """
    depth = np.asarray(mesh.points)[:, ax]
    proj_lo = s * (center_xyz[ax] - extent_xyz[ax] * SPAN_FRAC)
    proj_hi = s * (center_xyz[ax] + extent_xyz[ax] * SPAN_FRAC)
    pmax, pmin = max(proj_lo, proj_hi), min(proj_lo, proj_hi)
    t = np.clip((pmax - s * depth) / (pmax - pmin + 1e-9), 0, 1)
    return _DEPTH_LUT[(t * 255).astype(np.int64)]


def _mesh_view(pv, mesh, direction, up, center_xyz, extent_xyz, cell, scale, colors):
    """Render the (already depth-colored) mesh from the given camera direction."""
    pl = pv.Plotter(off_screen=True, window_size=(cell, cell))
    pl.set_background("white")
    pl.add_mesh(
        mesh,
        scalars=colors,
        rgb=True,
        opacity=MESH_OPACITY,
        smooth_shading=True,
        specular=0.3,
    )
    dist = max(extent_xyz) * 4
    pos = [center_xyz[j] + direction[j] * dist for j in range(3)]
    pl.camera_position = [tuple(pos), tuple(center_xyz), up]
    pl.enable_parallel_projection()
    pl.camera.parallel_scale = scale
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
    n_slices,
    seg_color_mode,
    mask_mode,
    mask_pad_nm,
    contrast_auto,
    fit,
):
    rscale, rtrans = raw_st
    sscale, strans = seg_st
    center_zyx = [center_xyz[2], center_xyz[1], center_xyz[0]]
    # Cubic field of view (same nm box in z/y/x) so every section -- and the mesh
    # views -- share one nm/pixel scale, regardless of the object's aspect ratio.
    fov_half = max(extent_xyz) / 2 * margin
    half_zyx = [fov_half, fov_half, fov_half]
    seg_vox_nm = min(sscale)  # isotropic seg voxel size; dilate in voxels
    pad_vox = int(round(mask_pad_nm / seg_vox_nm)) if mask_pad_nm else 0

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

    if contrast_auto:
        # Per-object window from the object's own raw voxels (nearest-resample the
        # seg mask onto the raw grid), applied to all slices -- stretches internal
        # texture while staying consistent within the object.
        idx = [
            (np.arange(raw_crop.shape[a]) * seg_crop.shape[a] // raw_crop.shape[a])
            for a in range(3)
        ]
        obj_vox = raw_crop[seg_crop[np.ix_(*idx)] == int(obj_id)]
        if obj_vox.size >= 16:
            lo, hi = (float(v) for v in np.percentile(obj_vox, [2, 98]))

    fsize = max(14, cell // 22)
    font = _font(fsize)
    have_mesh = mesh is not None
    # Each slice row gets a depth color (matches the mesh's depth coloring):
    # row 0 = nearest the camera (blue) ... last row = farthest (red).
    slice_colors = [_depth_color(i / (n_slices - 1)) for i in range(n_slices)]

    # Landscape layout: one ROW per axis. Columns read blue -> red across:
    # [front mesh (near/blue)] [slice1 .. sliceN] [back mesh (far/red)].
    ncol = n_slices + (2 if have_mesh else 0)
    border = max(3, cell // 90)
    canvas = Image.new(
        "RGB",
        (ncol * cell + (ncol + 1) * SEP, 3 * cell + 4 * SEP),
        SEP_COLOR,
    )

    for r, (label, axis, color) in enumerate(PLANES):
        y = SEP + r * (cell + SEP)
        ax = 2 - axis  # x,y,z depth-axis index for this row
        ext_nm = extent_xyz[ax]
        # Order slices near->far along the viewing direction (camera on +X/-Y/+Z).
        s = MESH_VIEW[axis][0][ax]
        fracs = (
            np.linspace(SPAN_FRAC, -SPAN_FRAC, n_slices)
            if s > 0
            else np.linspace(-SPAN_FRAC, SPAN_FRAC, n_slices)
        )
        # --fit view: shrink this row's window to its in-plane extent so the object
        # fills the panel (x/y/z scales then differ; sizes are still on the labels).
        inplane = [extent_xyz[i] for i in range(3) if i != ax]
        # Mesh views fill to the in-plane extent for view/panel; sections then crop
        # per-row (view) or per-slice (panel). cube keeps the shared cubic scale.
        disp_half = (
            min(max(inplane) / 2 * VIEW_PAD, fov_half)
            if fit in ("view", "panel")
            else fov_half
        )
        fit_frac = disp_half / fov_half
        col0 = 0
        back = None
        if have_mesh:
            direction, up = MESH_VIEW[axis]
            colors = _depth_colors(mesh, ax, s, center_xyz, extent_xyz)
            front_sign = "+" if direction[ax] > 0 else "-"
            back_sign = "-" if direction[ax] > 0 else "+"
            back_dir = tuple(-d for d in direction)
            # Front mesh (near/blue side) on the left, next to the blue slices.
            front = _bordered(
                _mesh_view(
                    pv,
                    mesh,
                    direction,
                    up,
                    center_xyz,
                    extent_xyz,
                    cell,
                    disp_half,
                    colors,
                ),
                color,
                border,
            )
            fd = ImageDraw.Draw(front)
            fd.text((8, 6), f"{front_sign}{label}", fill=color, font=font)
            fd.text(
                (8, 6 + fsize + 4),
                f"{label.lower()}_extent = {ext_nm:.0f} nm",
                fill=color,
                font=font,
            )
            canvas.paste(front, (SEP, y))
            # Back mesh (far/red side) -- built now, pasted after the slices so the
            # row reads continuously blue (front) -> ... -> red (back).
            # Mirror the back view left-right so its horizontal axis matches the
            # slices (and the front view): the 180-deg back camera flips handedness,
            # so flipping it back makes object positions line up across the row.
            back_img = _mesh_view(
                pv, mesh, back_dir, up, center_xyz, extent_xyz, cell, disp_half, colors
            ).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            back = _bordered(back_img, color, border)
            ImageDraw.Draw(back).text(
                (8, 6), f"{back_sign}{label}", fill=color, font=font
            )
            col0 = 1

        for i, frac in enumerate(fracs):
            pos = center_xyz[ax] + ext_nm * frac
            ri = int(round((pos - rtrans[axis]) / rscale[axis])) - rb[axis][0]
            si = int(round((pos - strans[axis]) / sscale[axis])) - sb[axis][0]
            tint = slice_colors[i] if seg_color_mode == "depth" else OVERLAY_RGB
            sec = _bordered(
                _section_at(
                    raw_crop,
                    seg_crop,
                    axis,
                    ri,
                    si,
                    obj_id,
                    cell,
                    lo,
                    hi,
                    tint,
                    mask_mode,
                    pad_vox,
                    fit,
                    fit_frac,
                ),
                slice_colors[i],
                border,
            )
            d = ImageDraw.Draw(sec)
            if not have_mesh and i == 0:
                d.text(
                    (8, 6),
                    f"{label.lower()}_extent = {ext_nm:.0f} nm",
                    fill=color,
                    font=font,
                )
            x = SEP + (col0 + i) * (cell + SEP)
            canvas.paste(sec, (x, y))

        if back is not None:
            xb = SEP + (col0 + n_slices) * (cell + SEP)
            canvas.paste(back, (xb, y))
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
    p.add_argument(
        "--slices", type=int, default=5, help="number of sections stepping through"
    )
    p.add_argument("--cell-size", type=int, default=420, help="pixels per panel")
    p.add_argument("--margin", type=float, default=1.5, help="crop padding factor")
    p.add_argument(
        "--fit",
        choices=["cube", "view", "panel"],
        default="panel",
        help="'panel' (default): crop each section to its own cross-section (max "
        "zoom; every panel independent). 'view': one scale per axis row. 'cube': one "
        "shared scale per object. Sizes are always on the extent labels.",
    )
    p.add_argument(
        "--contrast",
        default="84,152",
        help="fixed raw-EM intensity window 'lo,hi' (uint8) applied to every slice, "
        "or 'auto' for a per-object window from the object's own voxels",
    )
    p.add_argument(
        "--seg-color",
        choices=["fixed", "depth"],
        default="depth",
        help="object tint color: 'depth' (default) matches the slice's depth color "
        "(same as its border and the mesh); 'fixed' uses light red",
    )
    p.add_argument(
        "--mask",
        choices=["off", "object", "dim"],
        default="object",
        help="'object' (default) = keep raw EM only inside the segmentation (white "
        "outside) so the object is isolated; 'dim' = darken everything outside; "
        "'off' = translucent seg overlay over the full EM (keeps neighbor context)",
    )
    p.add_argument(
        "--mask-pad",
        type=float,
        default=250.0,
        help="nm of surrounding EM to keep around the object for object/dim masks "
        "(a context halo); fills the tight panel crop with EM rather than white. "
        "0 = tight mask",
    )
    args = p.parse_args(argv)

    contrast_auto = args.contrast.strip().lower() == "auto"
    lo, hi = (
        (0.0, 255.0) if contrast_auto else (float(x) for x in args.contrast.split(","))
    )

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
    written: list[dict] = []
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
                args.slices,
                args.seg_color,
                args.mask,
                args.mask_pad,
                contrast_auto,
                args.fit,
            )
            if grid is None:
                print(f"  skip {oid}: empty crop")
                continue
            grid.save(args.output_dir / f"{oid}.png")
            written.append({"object_id": oid, "image": f"{oid}.png", "meta": meta})
            done += 1
    finally:
        if mesh_source is not None:
            mesh_source.close()

    # Write a self-describing collection sidecar so an MCP host can load this dir
    # directly: it serves these montages as the object images and carries a legend.
    sidecar = {
        "collection_id": cid,
        "organelle_type": col.organelle_type,
        "image_legend": LAYOUT_LEGEND,
        "objects": written,
    }
    (args.output_dir / "objects.json").write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {done} montages + objects.json -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
