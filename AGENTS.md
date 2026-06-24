# AGENTS.md — driving the proofreader pipeline

This repo ships a **`proofreader`** MCP server (registered for Claude Code, VSCode,
Gemini CLI, and Codex; see the README). Use it to classify segmented objects
(mitochondria here) from rendered montages and produce a ranked review queue. The
server stores no model and does no inference — **you** do the visual judgment.

## Rules — read before you start (non-negotiable)

The entire point of this task is your **own, fresh** visual judgment on **every** object.
A run that skips the looking is worse than useless — it looks complete but is fabricated.
So:

1. **Look at every object's image, one at a time, before classifying it.** All of them —
   not a sample. View ONE image, classify it, then move to the next. Do **not** read or
   open images in batches / in parallel "to be efficient" — one image, one verdict, in
   sequence. No skipping, no extrapolating one verdict to "similar-looking" objects.
2. **Form each verdict only from the image you just viewed.** Do **not** read, load, or
   `import` any prior state — never open `ignore/slices_all/proofreader_state.json`,
   `*_backup*.json`, `ignore/runs/**`, or any existing verdicts/rationales. If you happen
   to see prior labels (e.g. in `list_objects` output), ignore them and judge independently.
3. **Record every verdict by calling the `classify_object` MCP tool DIRECTLY — one tool
   call per object.** The verdict must be a real MCP tool invocation made by you, not
   anything else. Specifically forbidden, no matter how it's framed:
   - writing or running a helper/wrapper script (Python or shell) that performs the
     classification — even "one object at a time", even if it `import`s
     `proofreader_mcp` and calls `classify_object`/storage itself. A script calling the
     function is **not** the MCP tool call; it bypasses the host and does not count.
   - writing to `proofreader_state.json` (or any state file) directly;
   - bulk-inserting, copying, or rewriting verdicts (e.g. relabeling `model_version`).
   If you catch yourself reaching for a script to "make classification easier", stop —
   that is the bypass. Call the tool. Bypassing it is not doing the task.
4. **Do not reuse another model's verdicts.** Each model's run must be its own pass. If a
   verdict you'd write happens to match a prior one, that's fine — but it must come from
   your own look, made through `classify_object`, not from copying.
5. **Report honestly.** State exactly how many objects you viewed and classified this run.
   Never claim you "analyzed all N at full resolution" unless you actually opened all N.
   If you stopped early or were interrupted, say so.

**Do not read the server source (`src/proofreader_mcp/**`), the tool scripts, the README,
`pyproject.toml`, or the MCP config to do this task.** You do not need them — only call the
MCP tools listed below. Do not invoke `classify_object` (or any tool) by importing the
module or running `uv run python ...`; use the MCP tool interface your client exposes.
Efficiency is not a goal here; faithful per-object judgment is. Reading 100+ images one at
a time is expected and correct.

## Task prompt (paste to an agent)

> Use the **proofreader** MCP server to evaluate the mitochondria in `ignore/slices_all`.
>
> 1. `load_collection` with path `ignore/slices_all`.
> 2. `get_image_legend` and `get_rubric` — follow them. The legend explains how to read
>    each object's montage; the rubric defines what counts as an error.
> 3. `list_objects` to get the object ids (ignore any prior labels shown there).
> 4. Go through **every** object **one at a time, in sequence** (do NOT read images in
>    batches/parallel). For each: first view its image —
>    `get_object_image(collection_id, object_id)` or open `ignore/slices_all/<object_id>.png`
>    directly — **then** call the `classify_object` MCP tool **directly** with:
>    `primary_label` (acceptable | likely_overmerged | likely_oversegmented | junk |
>    misclassified | uncertain), `confidence` 0–1, one-sentence `rationale` citing the
>    visual evidence you just saw, `recommended_action` (split | merge | discard | relabel |
>    none | human_review), optional `flags`, and **`model_version` = your own model name**.
>    One direct `classify_object` tool call per object. Do **not** write or run a helper
>    script (even one-at-a-time, even one that imports `proofreader_mcp`) to classify — a
>    script is not the MCP tool call. Do **not** read existing verdicts/state, read the
>    server source/README/config, build contact sheets, or combine objects into one
>    image — see the Rules above.
> 5. `rank_objects`, show the top of the queue, then `export_queue` (format json).
> 6. Archive the run (shell): `uv run python tools/export_run.py ignore/slices_all`
> 7. Report how many objects you actually viewed and classified this run.

## How to read each montage

Always call `get_image_legend` first — it returns the authoritative layout. In short:
3 rows (X/Y/Z axis). Each row, left→right: a depth-colored 3D mesh from the +axis, then
N raw-EM cross-sections stepping near→far through the object, then the mesh from the
−axis (mirrored). The object is faintly tinted in each section; a thin EM halo gives
context; mesh panels are labeled with the object's extent in nm. Judge **shape** from
the meshes (overmerge/oversegment) and **identity/validity** from the EM sections.

## Important

- **View ONE object montage at a time, at full resolution.** Each `<id>.png` is already
  a dense montage (2 meshes + ~5 EM sections per axis, 3 axes). Do **not** build contact
  sheets / grids that combine many objects into one image — downsampling makes the EM and
  mesh detail unreadable and your verdicts unreliable. Open each montage as-is (via
  `get_object_image` or the PNG on disk) and classify it before moving on. You can still
  process all objects; just don't shrink-and-stack them.
- **Set `model_version` to your actual model** (e.g. `gemini-2.5-pro`, `gpt-5.5`,
  `claude-opus-4-8`). This labels the review-page rows and the archive directory.
- **If your client can't display MCP image results, read the PNGs off disk** — they are
  plain files at `ignore/slices_all/<id>.png`.
- **The live state keeps one verdict per object** (latest wins). So run
  `export_run` *immediately after your pass*, before any other model classifies; each
  run is saved to `ignore/runs/<model>/<YYYY-MM-DD_HHMMSS>/` (verdicts.json + queue.csv +
  review.html). If the live state mixes models, the archive falls back to `mixed/`
  (or pass `--model NAME` to force the label).
- **Do not run two agents against the same collection at once.** Each client runs its
  own MCP process; both rewrite the whole `proofreader_state.json` (last-writer-wins, no
  locking), so concurrent runs clobber each other's verdicts. Run sequentially, or give
  each agent its own copy of the collection dir (`cp -r ignore/slices_all
  ignore/slices_all_<agent>`) so they have separate state.

## Tools

`load_collection`, `list_objects`, `get_object_image`, `get_image_legend`, `get_rubric`,
`classify_object`, `rank_objects`, `export_queue`, `record_review` (human verdicts).

## Regenerating inputs (optional)

Montages come from `tools/render_slices.py` (raw EM + segmentation + mesh, OME-Zarr +
precomputed mesh sources). The review page comes from `tools/make_review_page.py`. See
the README for those; classification does not require re-rendering.
