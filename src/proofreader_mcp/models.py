"""Core data types and label/action vocabularies.

These dataclasses are the persisted shapes. They are deliberately plain (stdlib
only) so the storage and ranking layers can be tested without the MCP SDK
installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class PrimaryLabel(str, Enum):
    ACCEPTABLE = "acceptable"
    LIKELY_OVERMERGED = "likely_overmerged"
    LIKELY_OVERSEGMENTED = "likely_oversegmented"
    JUNK = "junk"
    MISCLASSIFIED = "misclassified"
    UNCERTAIN = "uncertain"


class RecommendedAction(str, Enum):
    SPLIT = "split"
    MERGE = "merge"
    DISCARD = "discard"
    RELABEL = "relabel"
    NONE = "none"
    HUMAN_REVIEW = "human_review"


class ReviewVerdict(str, Enum):
    ACCEPT = "accept"  # the model's classification was correct
    REJECT = "reject"  # the model's classification was wrong
    CORRECT = "correct"  # human supplied a corrected label (see notes)


def _coerce_enum(enum_cls: type[Enum], value: Any) -> Enum:
    """Accept either an enum member or its string value."""
    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)


@dataclass
class ObjectRecord:
    """One object in a collection: its image plus optional upstream metadata."""

    object_id: str
    image: str  # filename relative to the collection directory
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ObjectRecord":
        return cls(
            object_id=str(d["object_id"]),
            image=str(d["image"]),
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class Verdict:
    """A classification produced by the host LLM for one object."""

    object_id: str
    primary_label: PrimaryLabel
    confidence: float
    rationale: str
    recommended_action: RecommendedAction
    flags: list[PrimaryLabel] = field(default_factory=list)
    image_hash: str | None = None
    model_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "primary_label": self.primary_label.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "recommended_action": self.recommended_action.value,
            "flags": [f.value for f in self.flags],
            "image_hash": self.image_hash,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Verdict":
        return cls(
            object_id=str(d["object_id"]),
            primary_label=_coerce_enum(PrimaryLabel, d["primary_label"]),
            confidence=float(d["confidence"]),
            rationale=str(d.get("rationale", "")),
            recommended_action=_coerce_enum(
                RecommendedAction, d.get("recommended_action", "none")
            ),
            flags=[_coerce_enum(PrimaryLabel, f) for f in d.get("flags") or []],
            image_hash=d.get("image_hash"),
            model_version=d.get("model_version"),
        )


@dataclass
class Review:
    """A human verdict on a model classification."""

    object_id: str
    verdict: ReviewVerdict
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "verdict": self.verdict.value,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Review":
        return cls(
            object_id=str(d["object_id"]),
            verdict=_coerce_enum(ReviewVerdict, d["verdict"]),
            notes=str(d.get("notes", "")),
        )
