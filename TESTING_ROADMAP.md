# Testing Roadmap: Mesh-Driven Macro Visual Proofreading

**Objective:** an autonomous *Scan → Target → Deep-Dive* loop on top of the
`proofreader-mcp` server. Instead of feeding the agent a fixed montage per object,
we hand it a **rotating 3D turntable video** of one object's mesh with a labeled
grid overlay, let it **choose** where the structural anomalies are (grid cell +
timestamp), translate those choices back to voxel coordinates, and serve raw EM /
segmentation sub-volumes there for a final SPLIT/MERGE verdict.

---

## Phase 1 — Local tooling & generation (deterministic backend) ✅ done

Everything an agent's choices get translated *through*. No agent involved; pure,
testable Python.

### Test 1.1 — 3D turntable video generation ✅

`tools/render_turntable.py` loads one target mesh, centers it in a **static**
wireframe cube, spins the mesh about the screen-vertical axis, and composites a
**stationary** labeled 2D grid overlay (columns `A..`, rows `1..`) on every frame.
Writes `<id>.mp4` plus an `<id>.turntable.json` sidecar describing the exact
camera, grid, rotation, and voxel scale.

```bash
uv sync --extra render --extra video --extra precomputed   # deps
xvfb-run -a uv run python tools/render_turntable.py \
    /path/to/mesh/multires 12345 /path/to/out \
    --source precomputed --voxel-size 8 8 8 --seconds 3 --fps 30
# -> /path/to/out/12345.mp4 + 12345.turntable.json
```

**Invariant that makes the whole loop work:** the camera, the wireframe box, and
the grid overlay are fixed in screen space; *only the mesh spins.* So a grid cell
always names the same screen region and the only per-frame variable is the mesh's
rotation angle. Projection is **orthographic**, which keeps the grid → ray mapping
linear and exact. Verified: smoke render produced a clean ~90 KB, 3 s/30 fps MP4
with a stationary grid and a visibly spinning mesh.

### Test 1.2 — Grid-to-voxel ray-cast mapping ✅

`tools/grid_to_voxel.py` reads the sidecar and inverts a visual hit. For
`{"grid": "D4", "timestamp": "00:02"}` it builds the fixed orthographic ray for
that cell, **undoes that frame's rotation** on the ray, intersects the *un-rotated*
mesh, and returns the absolute `[x, y, z]` voxel coordinate (plus nm and a `hit`
flag — a cell whose ray clears the object reports `hit: false`).

```bash
uv run python tools/grid_to_voxel.py /path/to/out/12345.turntable.json \
    --grid D4 --timestamp 00:02
# or batch (the array an agent emits in Phase 2):
uv run python tools/grid_to_voxel.py .../12345.turntable.json --hits-json hits.json
```

The geometry (`tools/turntable_geometry.py`) is pure numpy and unit-tested in
`tests/test_turntable.py` (19 tests): label/timestamp parsing, camera-basis
orthonormality, rotation-invariance of a cell's ray Z, and a closed-form
end-to-end check — a sphere is symmetric about the spin axis, so a grid hit's Z is
analytically fixed regardless of rotation, and the full forward+inverse pipeline
reproduces it. Pure-geometry tests need only numpy; the ray-cast tests need
pyvista (CPU `ray_trace`, no GL) and skip when it is absent (e.g. CI's `[dev]`).

---

## Phase 2 — Macro-scan & target localization (agent testing) ⏳ next

Test a vision-language agent's ability to ingest the video, use the visual grid, and emit
structured anomaly hits.

### Test 2.1 — Multi-anomaly JSON extraction

**Lesson learned:** agents localize poorly on the *video*. The API sub-samples it
to a few downscaled frames, so frame numbers and small grid labels are unreadable
and the model only forms a vague sense of the rotation — frame/cell answers came
back nonsensical. Fix: feed **labeled still viewpoints** instead
(`render_turntable.py --stills N`). The model sees every view at full resolution,
points approximately, and the deterministic ray-cast supplies the precision.

- **Action:** render `--stills 8`, pass the `<id>.view*.png` images to the agent with
  `prompts/turntable_scan.md` (defines views, coarse grid, pointing, anomaly
  vocabulary), demanding **only** a JSON array.
- **Success:** the array's `{view, x, y | cell, anomaly_type, confidence}` entries
  feed straight into `grid_to_voxel.py --hits-json` and resolve to on-mesh voxels
  (`hit: true`) with a local slice orientation.
- **TODO:**
  - [x] Stills mode (`--stills N`) — high-res labeled viewpoints, the agent-friendly input.
  - [x] `prompts/turntable_scan.md` — stills + pointing prompt; output feeds `grid_to_voxel.py`.
  - [x] `grid_to_voxel.py` accepts `{view, x/y point}` and returns a slice orientation
        (`surface_normal` + PCA `principal_axes`) for a meaningful cross-section.
  - [ ] Wire the agent CLI to accept the still images (confirm input path/size limits).
  - [ ] Eval harness: render N known-bad objects, score recall of seeded anomalies.

---

## Phase 3 — Dynamic MCP navigation (closing the loop) ⏳ later

Feed the resolved voxel targets back through the MCP server for a final verdict.

### Test 3.1 — Automated voxel deep-dive integration

- **Action:** pipe `grid_to_voxel.py` output into a new MCP tool that serves a
  parameterized raw-EM + segmentation sub-volume around each coordinate.
- **Success:** TensorStore query returns the correct sub-volume with no human step.
- **TODO:**
  - [ ] Add `get_volume_chunk(center_voxel, size, ...)` MCP tool (TensorStore-backed;
        `tools/render_slices.py` already reads OME-Zarr EM+seg and is the starting point).
  - [ ] Define the chunk request schema shared with `grid_to_voxel.py` output.

### Test 3.2 — Agentic Z-axis scrolling validation

- **Action:** give the agent `shift_slice_offset` and challenge it to verify an
  ambiguous membrane border by stepping through successive slices.
- **Success:** the agent queries neighboring slices, tracks membrane continuity, and
  outputs a definitive **SPLIT** or **MERGE** verdict.
- **TODO:**
  - [ ] Add `shift_slice_offset` MCP tool (stateful current-slice cursor per object).
  - [ ] Extend `classify_object` / rubric so a turntable-driven SPLIT/MERGE verdict is
        recordable alongside the existing montage-based labels.

---

## Status

| Phase | Item | State |
|-------|------|-------|
| 1.1 | Turntable video generation | ✅ `tools/render_turntable.py` |
| 1.2 | Grid-to-voxel ray-cast mapping | ✅ `tools/grid_to_voxel.py` + tests |
| 2.1 | Multi-anomaly JSON extraction | ⏳ scan prompt + agent wiring |
| 3.1 | Voxel deep-dive integration | ⏳ `get_volume_chunk` MCP tool |
| 3.2 | Agentic Z-axis scrolling | ⏳ `shift_slice_offset` MCP tool |
