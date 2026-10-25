#!/usr/bin/env python
"""Map a turntable grid hit ``{grid, timestamp}`` back to a voxel coordinate.

Phase 1, Test 1.2 of the mesh-driven visual proofreading roadmap. The agent
watches a turntable MP4 (see render_turntable.py) and reports visual hits as grid
cell + timestamp, e.g. ``{"grid": "D4", "timestamp": "00:02"}``. This tool reads
the matching ``<id>.turntable.json`` sidecar, reconstructs the exact ray that the
center of that grid cell traced into the scene at that moment, undoes the mesh's
spin for that frame, and intersects the *un-rotated* mesh — yielding the absolute
3D ``[x, y, z]`` voxel coordinate of the surface the agent pointed at.

Run with a single hit::

    uv run python tools/grid_to_voxel.py /path/to/out/12345.turntable.json \
        --grid D4 --timestamp 00:02

…or a batch (the JSON array an agent emits in Phase 2)::

    uv run python tools/grid_to_voxel.py /path/to/out/12345.turntable.json \
        --hits-json hits.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mesh_sources import make_source
from turntable_geometry import (
    TurntableGeometry,
    parse_grid_label,
    parse_timestamp,
    seconds_to_frame,
)


def _local_orientation(mesh, point: np.ndarray, cell_id: int | None, radius: float):
    """Local mesh frame at a surface hit, for orienting a meaningful slice.

    Returns the hit triangle's outward ``surface_normal`` plus the PCA of nearby
    surface vertices: ``principal_axes`` are the eigenvectors ordered
    largest->smallest eigenvalue. For a pinch/neck the largest axis runs *along*
    the bridge, so a slice plane with that axis as its normal shows the
    constriction's cross-section — the view that reveals a false merge. The
    smallest axis is the local surface/sheet normal. Downstream picks the basis
    that matches the anomaly type; we hand back the raw frame either way.
    """
    pts = np.asarray(mesh.points, float)
    d = np.linalg.norm(pts - point, axis=1)
    near = pts[d <= radius]
    if len(near) < 8:  # sparse region: fall back to the nearest k vertices
        near = pts[np.argsort(d)[: min(32, len(pts))]]
    cov = np.cov((near - near.mean(0)).T)
    evals, evecs = np.linalg.eigh(cov)  # ascending
    order = np.argsort(evals)[::-1]  # largest -> smallest
    axes = evecs[:, order].T

    info = {
        "principal_axes": [[round(float(x), 4) for x in a] for a in axes],
        "eigenvalues": [round(float(evals[i]), 2) for i in order],
        "neighbors": int(len(near)),
    }
    if cell_id is not None:
        try:
            n = np.asarray(mesh.cell_normals[cell_id], float)
            info["surface_normal"] = [round(float(x), 4) for x in n]
        except (IndexError, TypeError, ValueError):
            pass
    return info


def map_hit(
    geom: TurntableGeometry,
    mesh,
    grid: str | None = None,
    timestamp: str | float | None = None,
    frame: int | None = None,
    *,
    point: tuple[float, float] | None = None,
    view: int | None = None,
) -> dict:
    """Resolve one grid/point hit to nm + voxel coordinates + a slice orientation.

    The screen location is a coarse grid ``grid`` cell center, or a fine
    normalized ``point`` ``(x, y)`` in 0..1 (preferred — the model points more
    precisely than it names a cell). The frame is ``view`` or ``frame`` (the
    still/video index), else ``timestamp``. On a hit, the ray-cast snaps to the
    exact surface voxel and the local mesh geometry gives a meaningful slice
    orientation; ``hit`` is False when the ray misses the object.
    """
    if view is not None:
        frame = view
    if frame is not None:
        frame = max(0, min(geom.n_frames - 1, int(frame)))
    elif timestamp is not None:
        frame = seconds_to_frame(parse_timestamp(timestamp), geom.fps, geom.n_frames)
    else:
        raise ValueError("provide a frame, view, or timestamp")

    col = row = None
    if point is not None:
        u, v = float(point[0]), float(point[1])
    elif grid is not None:
        col, row = parse_grid_label(grid, geom.n_cols, geom.n_rows)
        u, v = geom.cell_center_fraction(col, row)
    else:
        raise ValueError("provide a grid cell or a normalized point (x, y)")

    start, stop = geom.frame_ray_uv(u, v, frame)
    points, ind = mesh.ray_trace(start, stop, first_point=True)
    points = np.asarray(points, float)

    result = {
        "grid": grid.strip().upper() if grid else None,
        "point": [round(u, 4), round(v, 4)] if point is not None else None,
        "timestamp": timestamp,
        "column": col,
        "row": row,
        "frame": frame,
        "angle_degrees": round(float(np.rad2deg(geom.frame_angle_rad(frame))), 3),
        "hit": points.size > 0,
    }
    if points.size > 0:
        voxel = geom.nm_to_voxel(points)
        result["point_nm"] = [round(float(v), 2) for v in points]
        result["voxel"] = [round(float(v), 3) for v in voxel]
        result["voxel_int"] = [int(round(float(v))) for v in voxel]
        cell_id = int(np.atleast_1d(ind)[0]) if np.asarray(ind).size else None
        result["orientation"] = _local_orientation(
            mesh, points, cell_id, radius=geom.box_half * 0.12
        )
    return result


def load_sidecar(path: Path) -> tuple[TurntableGeometry, dict]:
    data = json.loads(Path(path).read_text())
    return TurntableGeometry.from_dict(data["geometry"]), data


def _load_mesh(sidecar: dict):
    src = sidecar["source"]
    source = make_source(
        src["input"], kind=src.get("kind", "auto"), lod=src.get("lod", 0)
    )
    try:
        return source.get(sidecar["object_id"])
    finally:
        source.close()


def _hit_kwargs(h: dict) -> dict:
    """Normalize one hit dict (from an agent or the CLI) into map_hit kwargs.

    Accepts ``grid``/``cell`` for the coarse cell, ``point``/``[x, y]`` for a fine
    location, and ``view``/``frame``/``timestamp`` for the moment. Unknown extra
    fields (anomaly_type, confidence, description) are ignored.
    """
    point = h.get("point")
    if point is None and h.get("x") is not None and h.get("y") is not None:
        point = [h["x"], h["y"]]
    return {
        "grid": h.get("grid") or h.get("cell"),
        "point": point,
        "view": h.get("view"),
        "frame": h.get("frame"),
        "timestamp": h.get("timestamp"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar", type=Path, help="<id>.turntable.json from render")
    parser.add_argument("--grid", help="grid cell, e.g. D4")
    parser.add_argument(
        "--point",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="normalized location 0..1 (preferred over --grid; finer)",
    )
    parser.add_argument("--view", type=int, help="still view index")
    parser.add_argument("--frame", type=int, help="video frame index from the HUD")
    parser.add_argument("--timestamp", help="mm:ss or seconds (if no frame/view)")
    parser.add_argument(
        "--hits-json",
        type=Path,
        help="JSON array of hits, each {grid|x,y, view|frame|timestamp} (batch mode)",
    )
    args = parser.parse_args(argv)

    geom, sidecar = load_sidecar(args.sidecar)

    has_loc = args.grid or args.point
    has_time = args.view is not None or args.frame is not None or args.timestamp
    if args.hits_json:
        hits = json.loads(args.hits_json.read_text())
    elif has_loc and has_time:
        hits = [
            {
                "grid": args.grid,
                "x": args.point[0] if args.point else None,
                "y": args.point[1] if args.point else None,
                "view": args.view,
                "frame": args.frame,
                "timestamp": args.timestamp,
            }
        ]
    else:
        print(
            "provide a location (--grid or --point) and a frame "
            "(--view/--frame/--timestamp), or --hits-json",
            file=sys.stderr,
        )
        return 2

    try:
        mesh = _load_mesh(sidecar)
    except Exception as exc:  # noqa: BLE001 - surface backend errors cleanly
        print(f"could not load mesh: {exc}", file=sys.stderr)
        return 2

    results = [map_hit(geom, mesh, **_hit_kwargs(h)) for h in hits]
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
