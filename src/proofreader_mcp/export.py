"""Render a ranked review queue to CSV or JSON.

The queue is the human-facing deliverable: a priority-sorted table the
proofreader steps through.
"""

from __future__ import annotations

import csv
import io
import json

from .ranking import RankedItem

QUEUE_COLUMNS = [
    "object_id",
    "primary_label",
    "flags",
    "confidence",
    "severity",
    "expected_cost",
    "rationale",
    "recommended_action",
]


def to_csv(items: list[RankedItem]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=QUEUE_COLUMNS)
    writer.writeheader()
    for it in items:
        row = it.to_dict()
        row["flags"] = ";".join(row["flags"])  # flatten list for a flat table
        writer.writerow(row)
    return buf.getvalue()


def to_json(items: list[RankedItem]) -> str:
    return json.dumps([it.to_dict() for it in items], indent=2)


def export_queue(items: list[RankedItem], fmt: str = "csv") -> str:
    if fmt == "csv":
        return to_csv(items)
    if fmt == "json":
        return to_json(items)
    raise ValueError(f"unknown export format: {fmt!r} (expected 'csv' or 'json')")
