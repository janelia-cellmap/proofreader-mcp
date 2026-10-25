# Agent task prompt

Paste the block below to an agent (Claude Code / VSCode / Gemini CLI / Codex) to
run a proofreading pass. The substantive evaluation criteria are NOT in this
prompt by design — the agent fetches them live via `get_image_legend` and
`get_rubric`. Behavior rules live in [AGENTS.md](AGENTS.md). Replace
`<collection_dir>` with the directory of rendered montages to evaluate.

---

Use the **proofreader** MCP server to evaluate the objects in `<collection_dir>`.

1. `load_collection` with path `<collection_dir>`.
2. `get_image_legend` and `get_rubric` — follow them. The legend explains how to read each object's montage; the rubric defines what counts as an error.
3. `list_objects` to get the object ids (ignore any prior labels shown there).
4. Go through **every** object **one at a time, in sequence** (do NOT read images in batches/parallel). For each: first view its image — `get_object_image(collection_id, object_id)` or open `<collection_dir>/<object_id>.png` directly — **then** call the `classify_object` MCP tool **directly** with: `primary_label` (acceptable | likely_overmerged | likely_oversegmented | junk | misclassified | uncertain), `confidence` 0–1, one-sentence `rationale` citing the visual evidence you just saw, `recommended_action` (split | merge | discard | relabel | none | human_review), optional `flags`, and **`model_version` = your own model name**. One direct `classify_object` tool call per object. Do **not** write or run a helper script (even one-object-at-a-time, even one that imports `proofreader_mcp`) to do the classifying — a script is not the MCP tool call. Do **not** read existing verdicts/state, read the server source/README/config, build contact sheets, or combine objects into one image — see the Rules in AGENTS.md.
5. `rank_objects`, show the top of the queue, then `export_queue` (format json).
6. Archive the run (shell): `uv run python tools/export_run.py <collection_dir>`
7. Report how many objects you actually viewed and classified this run.
