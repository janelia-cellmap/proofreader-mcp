"""Deterministic geometry for the mesh-driven turntable proofreading loop.

This module is the shared, *pure* backend that ties the two halves of Phase 1
together:

* ``tools/render_turntable.py`` builds a rotating MP4 of one object's mesh,
  centered in a static wireframe cube with a labeled 2D grid overlay (columns
  A.., rows 1..), and writes a sidecar JSON describing exactly how it was
  rendered.
* ``tools/grid_to_voxel.py`` reads that sidecar and maps a
  ``{"grid": "D4", "timestamp": "00:02"}`` visual hit (the kind an agent emits
  after watching the video) back to an absolute ``[x, y, z]`` voxel coordinate
  in the data store, by ray-casting through the mesh.

The whole pipeline hinges on one invariant: **the camera, the wireframe box,
and the grid overlay are fixed in screen space; only the mesh spins.** So a grid
cell always names the same screen region, and the only thing that changes frame
to frame is the mesh's rotation angle. Mapping a (grid cell, timestamp) hit back
to 3D is therefore: build the fixed orthographic ray for that grid cell, undo
the frame's rotation on that ray, and intersect the *un-rotated* mesh.

Everything here is plain numpy so it is importable and unit-testable without
pyvista or a GL context; the ray/mesh intersection itself lives in
``grid_to_voxel.py`` where pyvista is available.

Conventions
-----------
* Coordinates are right-handed. The orthonormal camera basis is
  ``(right, up, forward)`` where ``forward`` points from the camera toward the
  object center and ``right = normalize(cross(up, forward))``.
* Screen fractions ``(u, v)`` run ``u`` 0..1 left->right and ``v`` 0..1
  top->bottom (image convention).
* Grid columns are lettered ``A, B, ...`` left->right; rows are numbered
  ``1, 2, ...`` top->bottom. So ``"D4"`` is the 4th column, 4th row.
* The mesh spins about ``up_axis`` (the screen-vertical world axis) through the
  object center. Frame 0 is 0 degrees; frame ``k`` is
  ``k / n_frames * total_degrees``.
* Mesh vertices are in nanometers (as produced by the precomputed/file mesh
  sources). A voxel coordinate is ``(p_nm - origin_nm) / voxel_size_nm``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Labels: grid cells and timestamps
# --------------------------------------------------------------------------


def column_label(index: int) -> str:
    """0-based column index -> spreadsheet-style letters (0->A, 25->Z, 26->AA)."""
    if index < 0:
        raise ValueError(f"column index must be >= 0, got {index}")
    letters = ""
    n = index
    while True:
        letters = chr(ord("A") + n % 26) + letters
        n = n // 26 - 1
        if n < 0:
            break
    return letters


def column_index(label: str) -> int:
    """Inverse of :func:`column_label` (``"A"`` -> 0, ``"AA"`` -> 26)."""
    label = label.strip().upper()
    if not label or not label.isalpha():
        raise ValueError(f"invalid column label: {label!r}")
    n = 0
    for ch in label:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def parse_grid_label(label: str, n_cols: int, n_rows: int) -> tuple[int, int]:
    """``"D4"`` -> ``(col_idx, row_idx)`` 0-based, validated against the grid size.

    Accepts an optional separator (``"D-4"``, ``"D 4"``). Rows are 1-based in the
    label (matching the overlay) and returned 0-based from the top.
    """
    m = re.fullmatch(r"\s*([A-Za-z]+)[\s\-_]*([0-9]+)\s*", label)
    if not m:
        raise ValueError(f"could not parse grid label {label!r} (expected e.g. 'D4')")
    col = column_index(m.group(1))
    row = int(m.group(2)) - 1
    if not (0 <= col < n_cols):
        raise ValueError(
            f"column {m.group(1)!r} (index {col}) out of range 0..{n_cols - 1}"
        )
    if not (0 <= row < n_rows):
        raise ValueError(f"row {m.group(2)!r} (index {row}) out of range 1..{n_rows}")
    return col, row


def parse_timestamp(value: str | float | int) -> float:
    """Parse a timestamp to seconds. Accepts ``"mm:ss"``, ``"ss"``, or a number.

    Fractional seconds are allowed (``"00:02.5"``, ``2.5``).
    """
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 2:
            raise ValueError(f"expected mm:ss timestamp, got {value!r}")
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return float(text)


def seconds_to_frame(seconds: float, fps: int, n_frames: int) -> int:
    """Nearest frame index for a time, clamped to ``[0, n_frames - 1]``."""
    frame = int(round(seconds * fps))
    return max(0, min(n_frames - 1, frame))


# --------------------------------------------------------------------------
# Linear algebra
# --------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("cannot normalize a zero vector")
    return v / n


def camera_basis(
    forward: np.ndarray, up: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal ``(right, up, forward)`` from a view direction and rough up.

    ``forward`` points from the camera toward the center. ``up`` is
    re-orthogonalized against ``forward`` so the basis is exactly orthonormal.
    """
    f = _unit(forward)
    right = _unit(np.cross(up, f))
    true_up = np.cross(f, right)
    return right, true_up, f


def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix about a (not-necessarily-unit) ``axis``."""
    a = _unit(axis)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    x, y, z = a
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    return c * np.eye(3) + s * cross + (1 - c) * np.outer(a, a)


# --------------------------------------------------------------------------
# The geometry record (serialized to / from the sidecar)
# --------------------------------------------------------------------------


@dataclass
class TurntableGeometry:
    """Everything needed to render a turntable and to invert grid hits.

    Distances are in nm. ``center`` is the rotation pivot and the focal point of
    the orthographic camera. ``parallel_scale`` is half the viewport *height* in
    nm (pyvista's camera convention); the width is ``parallel_scale * aspect``.
    """

    center: np.ndarray  # (3,) object center in nm
    forward: np.ndarray  # (3,) camera view direction (camera -> center)
    up_axis: np.ndarray  # (3,) screen-vertical world axis; also the spin axis
    parallel_scale: float  # half the viewport height, nm
    window: tuple[int, int]  # (width, height) in pixels
    n_cols: int
    n_rows: int
    fps: int
    n_frames: int
    total_degrees: float
    box_half: float  # half-size of the static wireframe cube, nm
    voxel_size_nm: np.ndarray  # (3,) nm per voxel
    origin_nm: np.ndarray = field(  # world nm at voxel (0,0,0)
        default_factory=lambda: np.zeros(3)
    )

    # -- derived ----------------------------------------------------------

    @property
    def aspect(self) -> float:
        return self.window[0] / self.window[1]

    @property
    def right(self) -> np.ndarray:
        return camera_basis(self.forward, self.up_axis)[0]

    @property
    def up(self) -> np.ndarray:
        return camera_basis(self.forward, self.up_axis)[1]

    @property
    def duration_seconds(self) -> float:
        return self.n_frames / self.fps

    def frame_angle_rad(self, frame: int) -> float:
        """Spin angle (radians) of the mesh at a given frame."""
        return np.deg2rad(frame / self.n_frames * self.total_degrees)

    # -- screen <-> world -------------------------------------------------

    def cell_center_fraction(self, col: int, row: int) -> tuple[float, float]:
        """Screen ``(u, v)`` of a grid cell center (u left->right, v top->bottom)."""
        u = (col + 0.5) / self.n_cols
        v = (row + 0.5) / self.n_rows
        return u, v

    def plane_point(self, u: float, v: float) -> np.ndarray:
        """World point on the focal plane (through center) at screen ``(u, v)``."""
        right, up, _ = camera_basis(self.forward, self.up_axis)
        half_w = self.parallel_scale * self.aspect
        half_h = self.parallel_scale
        h = (2 * u - 1) * half_w  # +right
        w = (1 - 2 * v) * half_h  # +up (v=0 is top)
        return np.asarray(self.center, float) + h * right + w * up

    def ortho_ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """Fixed (un-rotated) orthographic ray ``(start, stop)`` through ``(u, v)``.

        The ray is parallel to ``forward``; ``start`` sits well in front of the
        box and ``stop`` well behind it, so it brackets the whole object.
        """
        _, _, f = camera_basis(self.forward, self.up_axis)
        p = self.plane_point(u, v)
        reach = self.box_half * 4.0 + self.parallel_scale
        return p - f * reach, p + f * reach

    def frame_ray_uv(
        self, u: float, v: float, frame: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ray for screen point ``(u, v)`` at a frame, in the *mesh's own* frame.

        The mesh spins by ``frame_angle_rad`` about ``up_axis``; rather than
        rotate the mesh we apply the inverse rotation to the fixed ray, so the
        result can be intersected against the static (un-rotated) mesh. The
        intersection is then directly a mesh-local nm coordinate. ``(u, v)`` may
        be a coarse grid-cell center or a fine pointed location.
        """
        start, stop = self.ortho_ray(u, v)
        r_inv = rotation_matrix(self.up_axis, -self.frame_angle_rad(frame))
        c = np.asarray(self.center, float)
        return r_inv @ (start - c) + c, r_inv @ (stop - c) + c

    def frame_ray(
        self, col: int, row: int, frame: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ray for a grid *cell center* at a frame (see :meth:`frame_ray_uv`)."""
        u, v = self.cell_center_fraction(col, row)
        return self.frame_ray_uv(u, v, frame)

    def nm_to_voxel(self, p_nm: np.ndarray) -> np.ndarray:
        """World-nm point -> fractional voxel coordinate."""
        return (np.asarray(p_nm, float) - self.origin_nm) / self.voxel_size_nm

    # -- (de)serialization ------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "center_nm": list(map(float, self.center)),
            "forward": list(map(float, self.forward)),
            "up_axis": list(map(float, self.up_axis)),
            "parallel_scale_nm": float(self.parallel_scale),
            "window": [int(self.window[0]), int(self.window[1])],
            "n_cols": int(self.n_cols),
            "n_rows": int(self.n_rows),
            "fps": int(self.fps),
            "n_frames": int(self.n_frames),
            "total_degrees": float(self.total_degrees),
            "box_half_nm": float(self.box_half),
            "voxel_size_nm": list(map(float, self.voxel_size_nm)),
            "origin_nm": list(map(float, self.origin_nm)),
        }

    @classmethod
    def from_dict(cls, d: dict) -> TurntableGeometry:
        return cls(
            center=np.asarray(d["center_nm"], float),
            forward=np.asarray(d["forward"], float),
            up_axis=np.asarray(d["up_axis"], float),
            parallel_scale=float(d["parallel_scale_nm"]),
            window=(int(d["window"][0]), int(d["window"][1])),
            n_cols=int(d["n_cols"]),
            n_rows=int(d["n_rows"]),
            fps=int(d["fps"]),
            n_frames=int(d["n_frames"]),
            total_degrees=float(d["total_degrees"]),
            box_half=float(d["box_half_nm"]),
            voxel_size_nm=np.asarray(d["voxel_size_nm"], float),
            origin_nm=np.asarray(d.get("origin_nm", [0, 0, 0]), float),
        )
