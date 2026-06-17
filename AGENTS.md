# AGENTS.md — driving the proofreader pipeline

This repo ships a **`proofreader`** MCP server (registered for Claude Code, VSCode,
Gemini CLI, and Codex; see the README). Use it to classify segmented objects
(mitochondria here) from rendered montages and produce a ranked review queue. The
server stores no model and does no inference — **you** do the visual judgment.

## Task prompt (paste to an agent)

> Use the **proofreader** MCP server to evaluate the mitochondria in `ignore/slices_all`.
>
> 1. `load_collection` with path `ignore/slices_all`.
> 2. `get_image_legend` and `get_rubric` — follow them. The legend explains how to read
>    each object's montage; the rubric defines what counts as an error.
> 3. `list_objects` to get the object ids.
> 4. For **each** object, look at its image — `get_object_image(collection_id, object_id)`
>    or open `ignore/slices_all/<object_id>.png` directly — then `classify_object` with:
>    `primary_label` (acceptable | likely_overmerged | likely_oversegmented | junk |
>    misclassified | uncertain), `confidence` 0–1, one-sentence `rationale` citing the
>    visual evidence, `recommended_action` (split | merge | discard | relabel | none |
>    human_review), optional `flags`, and **`model_version` = your own model name**.
> 5. `rank_objects`, show the top of the queue, then `export_queue` (format json).
> 6. Archive the run (shell): `uv run python tools/export_run.py ignore/slices_all`

## How to read each montage

Always call `get_image_legend` first — it returns the authoritative layout. In short:
3 rows (X/Y/Z axis). Each row, left→right: a depth-colored 3D mesh from the +axis, then
N raw-EM cross-sections stepping near→far through the object, then the mesh from the
−axis (mirrored). The object is faintly tinted in each section; a thin EM halo gives
context; mesh panels are labeled with the object's extent in nm. Judge **shape** from
the meshes (overmerge/oversegment) and **identity/validity** from the EM sections.

## Important

- **Set `model_version` to your actual model** (e.g. `gemini-2.5-pro`, `gpt-5.5`,
  `claude-opus-4-8`). This labels the review-page rows and the archive directory.
- **If your client can't display MCP image results, read the PNGs off disk** — they are
  plain files at `ignore/slices_all/<id>.png`.
- **The live state keeps one verdict per object** (latest wins). So run
  `export_run` *immediately after your pass*, before any other model classifies; each
  run is saved to `ignore/runs/<model>/<YYYY-MM-DD_HHMMSS>/` (verdicts.json + queue.csv +
  review.html). If the live state mixes models, the archive falls back to `mixed/`
  (or pass `--model NAME` to force the label).

## Tools

`load_collection`, `list_objects`, `get_object_image`, `get_image_legend`, `get_rubric`,
`classify_object`, `rank_objects`, `export_queue`, `record_review` (human verdicts).

## Regenerating inputs (optional)

Montages come from `tools/render_slices.py` (raw EM + segmentation + mesh, OME-Zarr +
precomputed mesh sources). The review page comes from `tools/make_review_page.py`. See
the README for those; classification does not require re-rendering.
