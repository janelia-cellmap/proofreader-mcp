"""Tests for the turntable geometry + grid->voxel inversion (Phase 1).

The turntable tooling lives under ``tools/`` and depends on the ``[render]`` /
``[video]`` extras (numpy, pyvista). CI installs only ``[dev]``, so every test
here skips cleanly when those libraries are absent. The pure-geometry tests need
only numpy; the end-to-end ray-cast test additionally needs pyvista (its
``ray_trace`` is a CPU operation, so no GL/display is required).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from turntable_geometry import (  # noqa: E402
    TurntableGeometry,
    column_index,
    column_label,
    parse_grid_label,
    parse_timestamp,
    rotation_matrix,
    seconds_to_frame,
)


def make_geometry(**overrides) -> TurntableGeometry:
    """A fixed, easy-to-reason-about geometry for the inversion tests."""
    params = dict(
        center=np.array([100.0, 100.0, 100.0]),
        forward=np.array([0.0, -1.0, 0.0]),
        up_axis=np.array([0.0, 0.0, 1.0]),
        parallel_scale=50.0,
        window=(960, 640),  # aspect 1.5
        n_cols=12,
        n_rows=8,
        fps=30,
        n_frames=90,
        total_degrees=360.0,
        box_half=50.0,
        voxel_size_nm=np.array([10.0, 10.0, 10.0]),
    )
    params.update(overrides)
    return TurntableGeometry(**params)


# --------------------------------------------------------------------------
# Labels and timestamps
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "index,label", [(0, "A"), (3, "D"), (25, "Z"), (26, "AA"), (27, "AB")]
)
def test_column_label_roundtrip(index, label):
    assert column_label(index) == label
    assert column_index(label) == index


def test_parse_grid_label():
    assert parse_grid_label("D4", 12, 8) == (3, 3)
    assert parse_grid_label("a1", 12, 8) == (0, 0)
    assert parse_grid_label("L8", 12, 8) == (11, 7)
    assert parse_grid_label("D-4", 12, 8) == (3, 3)


def test_parse_grid_label_out_of_range():
    with pytest.raises(ValueError):
        parse_grid_label("Z4", 12, 8)  # column past 12
    with pytest.raises(ValueError):
        parse_grid_label("D9", 12, 8)  # row past 8


@pytest.mark.parametrize(
    "value,seconds", [("00:02", 2.0), ("1:30", 90.0), ("2", 2.0), (2.5, 2.5)]
)
def test_parse_timestamp(value, seconds):
    assert parse_timestamp(value) == seconds


def test_seconds_to_frame_clamps():
    assert seconds_to_frame(2.0, fps=30, n_frames=90) == 60
    assert seconds_to_frame(0.0, fps=30, n_frames=90) == 0
    assert seconds_to_frame(100.0, fps=30, n_frames=90) == 89  # clamped


# --------------------------------------------------------------------------
# Camera + rotation geometry (numpy only)
# --------------------------------------------------------------------------


def test_camera_basis_orthonormal():
    geom = make_geometry()
    right, up, fwd = (
        geom.right,
        geom.up,
        np.asarray(geom.forward) / np.linalg.norm(geom.forward),
    )
    for a in (right, up, fwd):
        assert np.isclose(np.linalg.norm(a), 1.0)
    assert np.isclose(np.dot(right, up), 0.0)
    assert np.isclose(np.dot(right, fwd), 0.0)
    # forward -Y, up +Z => right +X
    assert np.allclose(right, [1.0, 0.0, 0.0])


def test_rotation_matrix_is_orthonormal():
    r = rotation_matrix(np.array([0.0, 0.0, 1.0]), np.deg2rad(240.0))
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_center_cell_ray_passes_through_center_at_frame_0():
    geom = make_geometry()
    # The exact screen center (u=v=0.5) ray must pass through the object center
    # and run parallel to the view direction.
    start, stop = geom.ortho_ray(0.5, 0.5)
    direction = (stop - start) / np.linalg.norm(stop - start)
    assert np.allclose(direction, [0.0, -1.0, 0.0])
    # center lies on the line: (center - start) is collinear with the direction
    assert np.allclose(np.cross(geom.center - start, direction), 0.0)


def test_frame_ray_z_is_rotation_invariant():
    """Spin is about Z, so a cell's ray keeps a constant Z across all frames."""
    geom = make_geometry()
    z_at_frame0 = geom.frame_ray(3, 3, 0)[0][2]
    for frame in (0, 30, 60, 89):
        start, stop = geom.frame_ray(3, 3, frame)
        assert np.isclose(start[2], z_at_frame0)
        assert np.isclose(stop[2], z_at_frame0)


def test_geometry_dict_roundtrip():
    geom = make_geometry()
    restored = TurntableGeometry.from_dict(geom.to_dict())
    assert np.allclose(restored.center, geom.center)
    assert np.allclose(restored.voxel_size_nm, geom.voxel_size_nm)
    assert restored.n_cols == geom.n_cols
    assert restored.n_frames == geom.n_frames


# --------------------------------------------------------------------------
# End-to-end: grid hit -> voxel via ray-cast against a real mesh
# --------------------------------------------------------------------------


def test_grid_hit_to_voxel_sphere():
    """{'grid': 'D4', 'timestamp': '00:02'} resolves to a stable voxel coord.

    A sphere is rotationally symmetric about the spin (Z) axis, so the hit's Z is
    fixed analytically regardless of the frame's rotation: it equals the cell's
    screen-vertical world coordinate. That makes the full forward+inverse pipeline
    checkable against a closed-form value.
    """
    pv = pytest.importorskip("pyvista")
    from grid_to_voxel import map_hit  # noqa: E402

    geom = make_geometry()
    sphere = pv.Sphere(radius=40.0, center=tuple(geom.center))

    result = map_hit(geom, sphere, "D4", "00:02")

    assert result["hit"] is True
    assert result["frame"] == 60
    assert np.isclose(result["angle_degrees"], 240.0)

    # Row 4 of 8 -> v=(3+0.5)/8=0.4375 -> world up offset (1-2v)*parallel_scale.
    v = (3 + 0.5) / geom.n_rows
    expected_z_nm = geom.center[2] + (1 - 2 * v) * geom.parallel_scale
    expected_voxel_z = expected_z_nm / geom.voxel_size_nm[2]
    assert np.isclose(result["voxel"][2], expected_voxel_z, atol=1e-2)


def test_grid_hit_miss_returns_no_voxel():
    """A corner cell whose ray clears the object reports hit=False."""
    pv = pytest.importorskip("pyvista")
    from grid_to_voxel import map_hit  # noqa: E402

    geom = make_geometry()
    sphere = pv.Sphere(radius=20.0, center=tuple(geom.center))  # small, in a big box
    result = map_hit(geom, sphere, "A1", "00:00")  # top-left corner, far from center
    assert result["hit"] is False
    assert "voxel" not in result
