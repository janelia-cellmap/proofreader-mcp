# proofreader-mcp — v1 Design Spec

## Goal
Given thousands of per-object mesh screenshots, identify the objects most likely to
contain errors and present them to a human proofreader in priority order. Errors include
segmentation faults (overmerge, oversegment) *and* object-validity faults (junk/debris,
misclassified — e.g. an object labeled mitochondrion that isn't a plausible one).

## V1 Scope
Image-centric proofreading only. The primary input is **one composite snapshot per
object** — a single PNG showing multiple viewpoints (e.g. front / side / top / one
perspective) of just that object. The host LLM reasons over that one image; the MCP
server organizes images, exposes them as resources, and provides tools to classify and
rank.

Voxel volumes, mesh topology, morphology stats, and Neuroglancer integration are
explicitly deferred — but the architecture leaves seams for them (optional metadata,
per-organelle rubrics, the validity-flag path).

## Workflow (one object at a time)
1. `load_collection` → register a directory of per-object composite PNGs.
2. Iterate `list_objects`; for each, the host fetches the one image and calls
   `classify_object` with a label + flags + confidence + rationale + action.
3. `rank_objects` orders the verdicts by expected error cost.
4. `export_queue` emits a sorted review queue; `record_review` captures human verdicts.

No two-stage shortlist is needed — one composite image per object already carries the
multiview information, and each classify call is one cheap image. Thousands of objects =
thousands of independent, parallelizable calls.

## What's visible from a single object's image
- **Overmerge** — reliable. Implausible topology, two distinct structures fused, multiple
  cell bodies in one label.
- **Oversegment** — partially visible. Tell: a harsh **planar truncation face** mid-
  structure where the object is cut flat instead of ending naturally.
  - **Trap:** the same flat cut appears when an object is legitimately clipped by the
    dataset / field-of-view boundary. Rule: a cut *away from* any volume edge is
    suspicious; a cut *at* the edge is expected. The render must convey volume-edge
    position (bounding-box hint in the montage, or an `at_volume_edge` metadata flag),
    or the model can't separate these.
- **Junk / debris** — visible. Sub-threshold specks, disconnected dust, obvious artifacts.
- **Misclassified / not-valid-{type}** — visible against a per-organelle rubric. Real-
  looking object whose morphology doesn't match its assigned organelle type.

## Input contract
One composite PNG per object, plus an optional sidecar `objects.json`:
```json
{
  "collection_id": "jrc_mus-liver-zon-1_run42",
  "organelle_type": "mitochondrion",
  "objects": [
    {
      "object_id": "12345",
      "image": "12345.png",
      "meta": {
        "voxel_count": 84213,
        "at_volume_edge": false,
        "centroid": [x, y, z]
      }
    }
  ]
}
```
- `image` filename convention fallback: `{object_id}.png`.
- `organelle_type` selects the validity rubric (see below). If absent, the validity
  classes are skipped and only structural classes apply.
- `meta` is optional; `voxel_count` and `at_volume_edge` improve validity and oversegment
  judgments when present, but are never required.

**Render requirements (upstream pipeline must satisfy):** consistent viewpoints across
objects, neutral background, consistent scale, and a volume-edge cue if oversegment
detection matters.

## Classification schema
Per object the LLM produces:
- `primary_label` ∈ {
    `acceptable`,
    `likely_overmerged`,
    `likely_oversegmented`,
    `junk`,
    `misclassified`,
    `uncertain`
  }
- `flags` — optional array of secondary observations from the same set (e.g. an
  overmerged object with a junk fragment → primary `likely_overmerged`, flag `junk`)
- `confidence` ∈ [0, 1]
- `rationale` — one or two sentences naming the visual evidence
- `recommended_action` ∈ { `split`, `merge`, `discard`, `relabel`, `none`, `human_review` }

Persist each verdict with the image hash + model version so re-runs are comparable.

## Per-organelle validity rubric
The validity classes (`junk`, `misclassified`) are judged against a short rubric injected
into the prompt for the declared `organelle_type`. The framework is generic; mito is the
first rubric.

**Mitochondrion (v1 rubric):**
- *Valid:* roughly tubular or ellipsoidal, may be elongated or branched networks, smooth
  surface at mesh resolution, plausible size.
- *Misclassified:* flat sheet-like (more like ER/membrane), or spherical/compact with no
  elongation (possible other organelle).
- *Junk:* tiny isolated blob below plausible size, disconnected speckle, angular/boxy
  artifact. Size best confirmed via `meta.voxel_count` rather than by eye.

## Ranking objective
Priority = expected error cost ≈ `confidence × severity`.

`severity` is a configurable per-label weight, larger when the error is costlier to miss.
Suggested defaults: overmerge > oversegment > misclassified > junk; `acceptable` = 0.
Falls back to `confidence` alone when no weights/meta available.

## MCP surface
Resources:
- `collection://{collection_id}` — object index + organelle_type
- `object://{collection_id}/{object_id}` — the one composite image as an image content block

Tools:
- `load_collection(path)` → register a collection; report missing images / orphan files
- `list_objects(collection_id, page)` → paginated object IDs + meta
- `get_object_image(collection_id, object_id)` → the composite image block
- `classify_object(collection_id, object_id, primary_label, flags, confidence, rationale, recommended_action)`
- `rank_objects(collection_id, by="expected_cost")`
- `export_queue(collection_id, format="csv"|"json")`
- `record_review(collection_id, object_id, verdict, notes)`

Division of labor: the host LLM does the visual judgment; the server stores no model and
makes no inference calls — it serves images and persists classifications, rankings, and
human verdicts.

## Output and human loop
- `export_queue` emits a priority-sorted table: `object_id, primary_label, flags,
  confidence, severity, expected_cost, rationale, recommended_action`.
- The proofreader steps through it; `record_review` captures verdict
  (accept / reject / correct) + notes.
- Reviews measure precision per label and tune severity weights. This loop ships in V1 —
  without it you can't tell if the system works.

## Out of scope for V1 (future phases)
- Mesh geometry / topology analysis
- Morphology statistics
- Voxel segmentation volumes, raw EM, label volumes
- Neuroglancer state integration
- Neighbor/context renders (would unlock fuller oversegment recall)
- Automated correction beyond the coarse `recommended_action`

## Non-goals / constraints
- The server stores no model and no inference logic — it's a resource + state layer for
  the host LLM.
- No dependency on voxel data, EM, or Neuroglancer in V1.
- Verdicts must be reproducible (image hash + model version recorded).
