[![tests](https://github.com/janelia-cellmap/proofreader-mcp/actions/workflows/tests.yml/badge.svg)](https://github.com/janelia-cellmap/proofreader-mcp/actions/workflows/tests.yml)

# proofreader-mcp

An [MCP](https://modelcontextprotocol.io) server for **image-centric proofreading of
segmentation objects**. Given thousands of per-object mesh screenshots, it helps a host
LLM identify the objects most likely to contain errors and present them to a human
proofreader in priority order.

Errors include segmentation faults (**overmerge**, **oversegment**) *and* object-validity
faults (**junk/debris**, **misclassified** — e.g. an object labeled mitochondrion that
isn't a plausible one).

## Division of labor

The server stores **no model** and makes **no inference calls**. It is a resource + state
layer: it organizes images, exposes them as MCP resources, and persists classifications,
rankings, and human verdicts. The host LLM does all of the visual judgment.

## Input contract

One composite PNG per object (a single image showing multiple viewpoints of just that
object), in a directory, plus an optional sidecar `objects.json`:

```json
{
  "collection_id": "jrc_mus-liver-zon-1_run42",
  "organelle_type": "mitochondrion",
  "objects": [
    {
      "object_id": "12345",
      "image": "12345.png",
      "meta": { "voxel_count": 84213, "at_volume_edge": false, "centroid": [0, 0, 0] }
    }
  ]
}
```

- `image` filename convention fallback: `{object_id}.png`.
- `organelle_type` selects the validity rubric. If absent, validity classes are skipped and
  only structural classes apply.
- `meta` is optional; `voxel_count` and `at_volume_edge` improve validity and oversegment
  judgments when present, but are never required.

If no `objects.json` is present, the collection is built by scanning the directory for
`*.png` files, using each filename stem as the `object_id`.

## Workflow

1. `load_collection(path)` — register a directory of per-object composite PNGs.
2. Iterate `list_objects`; for each, fetch the image and call `classify_object` with a
   label + flags + confidence + rationale + action.
3. `rank_objects` orders the verdicts by expected error cost.
4. `export_queue` emits a sorted review queue; `record_review` captures human verdicts.

## MCP surface

**Resources**

- `collection://{collection_id}` — object index + organelle_type + rubric
- `object://{collection_id}/{object_id}` — the composite image as an image content block

**Tools**

- `load_collection(path)` — register a collection; reports missing images / orphan files
- `list_objects(collection_id, page, page_size)` — paginated object IDs + meta + verdict status
- `get_object_image(collection_id, object_id)` — the composite image block
- `get_rubric(organelle_type)` — the validity rubric to inject into the prompt
- `classify_object(collection_id, object_id, primary_label, confidence, rationale, recommended_action, flags, model_version)`
- `rank_objects(collection_id, by)` — order verdicts by expected cost
- `export_queue(collection_id, format)` — emit a priority-sorted queue (`csv` or `json`)
- `record_review(collection_id, object_id, verdict, notes)` — capture a human verdict

## Classification schema

- `primary_label` ∈ {`acceptable`, `likely_overmerged`, `likely_oversegmented`, `junk`,
  `misclassified`, `uncertain`}
- `flags` — optional secondary observations from the same set
- `confidence` ∈ [0, 1]
- `rationale` — one or two sentences naming the visual evidence
- `recommended_action` ∈ {`split`, `merge`, `discard`, `relabel`, `none`, `human_review`}

Each verdict is persisted with the **image hash + model version** so re-runs are
comparable.

## Ranking

Priority = expected error cost ≈ `confidence × severity`, where `severity` is a
configurable per-label weight (default ordering: overmerge > oversegment > misclassified >
junk; `acceptable` = 0). Falls back to `confidence` alone when no weights apply.

## State

Per-collection state (verdicts + human reviews) is persisted to `proofreader_state.json`
inside the collection directory, keyed by object id and image hash.

## Install & run

This project uses [uv](https://docs.astral.sh/uv/). The committed `uv.lock` pins the full
dependency set for reproducible environments.

```bash
uv sync --extra dev    # create .venv and install the project + dev tools
uv run proofreader-mcp # run the server over stdio
uv run pytest          # run the test suite
```

(Plain pip also works, since `pyproject.toml` is standard PEP 621:
`pip install -e ".[dev]"` then `proofreader-mcp`.)

Register with an MCP host (e.g. Claude Desktop / Claude Code) — point it at the uv-managed
entry point:

```json
{
  "mcpServers": {
    "proofreader": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/proofreader-mcp", "proofreader-mcp"]
    }
  }
}
```

## Generating the images (prototype renderer)

The server consumes one composite PNG per object; producing those from meshes is an
**upstream** concern. A prototype renderer lives in [tools/](tools/) (outside the installed
package, behind the `render` + `precomputed` optional extras) so you can test the full loop.
It turns meshes into `{object_id}.png` montages plus an `objects.json` sidecar, from either
of two sources (auto-detected, or forced with `--source`):

```bash
uv sync --extra render --extra precomputed

# 1) a directory of {object_id}.ply (or .obj/.stl) files:
uv run python tools/render_meshes.py /path/to/plys /path/to/out \
    --organelle-type mitochondrion --cell-size 512

# 2) a Neuroglancer precomputed mesh layer (sharded/unsharded, multi-LOD Draco),
#    local or remote (gs://, https://). Choose the level of detail with --lod:
uv run python tools/render_meshes.py /path/to/mesh/multires /path/to/out \
    --source precomputed --lod 1 --organelle-type mitochondrion --limit 50
```

Each PNG is a 2×3 montage of the six axis-aligned views, laid out so each axis is a column
with the + view on top and the − view beneath it:

```
+X +Y +Z
-X -Y -Z
```

Each column is annotated with that axis's bounding-box extent in nm (mesh coordinates are
already in nm); the full `[dx, dy, dz]` is also written to the sidecar as `meta.extent_nm`.
`--scale-mode global` (default) renders every object at the same world scale so size is
comparable by eye. For precomputed layers, `--lod 0` is finest and higher is
coarser/faster (it falls back to a coarser LOD if the requested one is missing); object ids
come from the layer's `segment_properties`, or pass `--ids` explicitly. Headless GL on a
cluster needs OSMesa/EGL or a virtual framebuffer (`xvfb-run`).

The mesh-loading backends live in [tools/mesh_sources.py](tools/mesh_sources.py)
(`FileMeshSource`, `PrecomputedMeshSource`), so new formats are easy to add.

This renderer is a **prototype destined for
[cellmap-analyze](https://github.com/janelia-cellmap/cellmap-analyze)**, where dask can
scale it and segmentation access can fill in `voxel_count` / `at_volume_edge` / `centroid`.
It is intentionally not part of the core MCP install.

## Scope (v1)

Image-centric proofreading only. Voxel volumes, mesh topology, morphology stats, and
Neuroglancer integration are explicitly deferred; the architecture leaves seams for them
(optional metadata, per-organelle rubrics, the validity-flag path). See
[proofreader-mcp-design.md](proofreader-mcp-design.md) for the full design spec.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
