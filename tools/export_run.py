#!/usr/bin/env python
"""Archive the current evaluation of a collection as a timestamped run.

The MCP keeps a single live ``proofreader_state.json`` per collection (one verdict
per object). This tool snapshots that state into

    <runs-root>/<model>/<YYYY-MM-DD_HHMMSS>/
        verdicts.json   # ranked verdicts + model
        queue.csv       # priority-sorted review queue
        review.html     # the sortable review page (thumbnails -> the montages)

so successive runs (different models, or the same model over time) are kept side
by side for comparison instead of overwriting each other.

Usage::

    uv run python tools/export_run.py /path/to/collection
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import make_review_page
from proofreader_mcp.export import export_queue
from proofreader_mcp.ranking import rank
from proofreader_mcp.storage import Store


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("collection_dir", type=Path, help="collection with verdicts")
    p.add_argument("--runs-root", type=Path, default=Path("runs"), help="archive root")
    p.add_argument(
        "--model", default=None, help="label for this run (default: from verdicts)"
    )
    p.add_argument(
        "--timestamp", default=None, help="override the YYYY-MM-DD_HHMMSS stamp"
    )
    p.add_argument(
        "--thumbnails",
        type=Path,
        default=None,
        help="dir of {id}.png thumbnails (default: the collection itself)",
    )
    args = p.parse_args(argv)

    store = Store()
    rep = store.load_collection(args.collection_dir)
    col = store.get(rep.collection_id)
    if not col.verdicts:
        print("no verdicts to export; classify the collection first")
        return 1

    ranked = rank(list(col.verdicts.values()))
    models = {v.model_version or "unknown" for v in col.verdicts.values()}
    model = args.model or (next(iter(models)) if len(models) == 1 else "mixed")
    stamp = args.timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Sanitize the model into a safe directory name.
    safe_model = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in model)
    run_dir = args.runs_root / safe_model / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    records = [
        {**it.to_dict(), "model": col.verdicts[it.object_id].model_version}
        for it in ranked
    ]
    (run_dir / "verdicts.json").write_text(
        json.dumps(
            {
                "collection_id": rep.collection_id,
                "model": model,
                "timestamp": stamp,
                "count": len(records),
                "verdicts": records,
            },
            indent=2,
        )
    )
    (run_dir / "queue.csv").write_text(export_queue(ranked, "csv"))

    thumbs = args.thumbnails or args.collection_dir
    make_review_page.main(
        [
            str(args.collection_dir),
            "--output",
            str(run_dir / "review.html"),
            "--thumbnails",
            str(thumbs),
            "--title",
            f"proofreader — {rep.collection_id} — {model} — {stamp} ({len(records)})",
        ]
    )
    print(f"archived run -> {run_dir}  ({len(records)} verdicts, model={model})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
