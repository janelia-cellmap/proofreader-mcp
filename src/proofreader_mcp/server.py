"""The MCP server: resources + tools over a :class:`Store`.

The server is a thin adapter. It owns no judgment: it serves images and rubrics,
records the host LLM's classifications, derives rankings from them, and persists
human reviews. All domain logic lives in the sibling modules.
"""

from __future__ import annotations

import json
import mimetypes
from typing import Any

from mcp.server.fastmcp import FastMCP, Image

from .export import export_queue as render_queue
from .models import (
    PrimaryLabel,
    RecommendedAction,
    Review,
    ReviewVerdict,
    Verdict,
)
from .ranking import rank
from .storage import CollectionError, Store

mcp = FastMCP("proofreader")
store = Store()


# --------------------------------------------------------------------------
# Resources
# --------------------------------------------------------------------------


@mcp.resource("collection://{collection_id}")
def collection_resource(collection_id: str) -> str:
    """The object index + organelle_type + validity rubric for a collection."""
    col = store.get(collection_id)
    listing = store.list_objects(collection_id, page=0, page_size=10_000)
    return json.dumps(
        {
            "collection_id": col.collection_id,
            "organelle_type": col.organelle_type,
            "rubric": store.rubric(collection_id),
            "total": listing["total"],
            "objects": listing["objects"],
        },
        indent=2,
    )


@mcp.resource("object://{collection_id}/{object_id}")
def object_image_resource(collection_id: str, object_id: str) -> Image:
    """The one composite image for an object, as an image content block."""
    col = store.get(collection_id)
    path = col.image_path(object_id)
    if not path.exists():
        raise CollectionError(f"image missing on disk for object '{object_id}': {path}")
    return Image(path=str(path))


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


@mcp.tool()
def load_collection(path: str) -> dict[str, Any]:
    """Register a directory of per-object composite PNGs as a collection.

    Reads an optional ``objects.json`` sidecar; otherwise scans ``*.png``.
    Reports missing images (declared but absent) and orphan files (PNG on disk
    not declared in the sidecar).
    """
    return store.load_collection(path).to_dict()


@mcp.tool()
def list_objects(
    collection_id: str, page: int = 0, page_size: int = 100
) -> dict[str, Any]:
    """Paginated object ids + meta + classification/review status."""
    return store.list_objects(collection_id, page=page, page_size=page_size)


@mcp.tool()
def get_object_image(collection_id: str, object_id: str) -> Image:
    """Return the composite image for one object as an image content block."""
    col = store.get(collection_id)
    path = col.image_path(object_id)
    if not path.exists():
        raise CollectionError(f"image missing on disk for object '{object_id}': {path}")
    return Image(path=str(path))


@mcp.tool()
def get_rubric(collection_id: str) -> dict[str, Any]:
    """Return the structural guidance + organelle validity rubric to inject."""
    col = store.get(collection_id)
    return {
        "collection_id": collection_id,
        "organelle_type": col.organelle_type,
        "rubric": store.rubric(collection_id),
    }


@mcp.tool()
def classify_object(
    collection_id: str,
    object_id: str,
    primary_label: str,
    confidence: float,
    rationale: str,
    recommended_action: str = "none",
    flags: list[str] | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Record the host LLM's classification of one object.

    ``primary_label`` ∈ {acceptable, likely_overmerged, likely_oversegmented,
    junk, misclassified, uncertain}; ``recommended_action`` ∈ {split, merge,
    discard, relabel, none, human_review}. The verdict is persisted with the
    image hash and model version so re-runs are comparable.
    """
    try:
        label = PrimaryLabel(primary_label)
        action = RecommendedAction(recommended_action)
        flag_enums = [PrimaryLabel(f) for f in (flags or [])]
    except ValueError as exc:
        raise CollectionError(str(exc)) from exc

    verdict = Verdict(
        object_id=object_id,
        primary_label=label,
        confidence=confidence,
        rationale=rationale,
        recommended_action=action,
        flags=flag_enums,
        model_version=model_version,
    )
    stored = store.add_verdict(collection_id, verdict)
    return stored.to_dict()


@mcp.tool()
def rank_objects(collection_id: str, by: str = "expected_cost") -> dict[str, Any]:
    """Order all recorded verdicts by expected error cost (confidence × severity).

    ``by`` is ``"expected_cost"`` (default) or ``"confidence"``.
    """
    col = store.get(collection_id)
    ranked = rank(list(col.verdicts.values()), by=by)
    return {
        "collection_id": collection_id,
        "by": by,
        "count": len(ranked),
        "ranked": [it.to_dict() for it in ranked],
    }


@mcp.tool()
def export_queue(collection_id: str, format: str = "csv") -> str:
    """Emit the priority-sorted review queue as CSV or JSON text."""
    col = store.get(collection_id)
    ranked = rank(list(col.verdicts.values()), by="expected_cost")
    return render_queue(ranked, fmt=format)


@mcp.tool()
def record_review(
    collection_id: str, object_id: str, verdict: str, notes: str = ""
) -> dict[str, Any]:
    """Capture a human verdict on a model classification.

    ``verdict`` ∈ {accept, reject, correct}. For ``correct``, put the corrected
    label and reasoning in ``notes``.
    """
    try:
        rv = ReviewVerdict(verdict)
    except ValueError as exc:
        raise CollectionError(str(exc)) from exc
    review = Review(object_id=object_id, verdict=rv, notes=notes)
    stored = store.add_review(collection_id, review)
    return stored.to_dict()


def main() -> None:
    """Console-script entry point: run the server over stdio."""
    # Make sure .png resolves to image/png on hosts with a sparse mime registry.
    mimetypes.add_type("image/png", ".png")
    mcp.run()


if __name__ == "__main__":
    main()
