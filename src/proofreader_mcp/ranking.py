"""Ranking: order verdicts by expected error cost.

Priority = expected error cost ~= confidence * severity, where severity is a
configurable per-label weight, larger when the error is costlier to miss.
Default ordering: overmerge > oversegment > misclassified > junk; acceptable = 0.
Falls back to confidence alone when no weights apply.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import PrimaryLabel, Verdict

# Default severity weights. acceptable contributes no cost; uncertain gets a
# moderate weight so genuinely ambiguous objects still surface for human review.
DEFAULT_SEVERITY: dict[PrimaryLabel, float] = {
    PrimaryLabel.LIKELY_OVERMERGED: 1.0,
    PrimaryLabel.LIKELY_OVERSEGMENTED: 0.8,
    PrimaryLabel.MISCLASSIFIED: 0.5,
    PrimaryLabel.JUNK: 0.3,
    PrimaryLabel.UNCERTAIN: 0.4,
    PrimaryLabel.ACCEPTABLE: 0.0,
}


@dataclass
class RankedItem:
    object_id: str
    primary_label: PrimaryLabel
    flags: list[PrimaryLabel]
    confidence: float
    severity: float
    expected_cost: float
    rationale: str
    recommended_action: str

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "primary_label": self.primary_label.value,
            "flags": [f.value for f in self.flags],
            "confidence": self.confidence,
            "severity": self.severity,
            "expected_cost": self.expected_cost,
            "rationale": self.rationale,
            "recommended_action": self.recommended_action,
        }


def expected_cost(
    verdict: Verdict, severity: dict[PrimaryLabel, float] | None = None
) -> float:
    weights = severity or DEFAULT_SEVERITY
    sev = weights.get(verdict.primary_label, 0.0)
    return verdict.confidence * sev


def rank(
    verdicts: list[Verdict],
    by: str = "expected_cost",
    severity: dict[PrimaryLabel, float] | None = None,
) -> list[RankedItem]:
    """Return verdicts ordered most-to-least worth a human's attention.

    ``by="expected_cost"`` (default) sorts by confidence * severity.
    ``by="confidence"`` falls back to raw confidence, ignoring severity.
    Ties break by confidence, then object_id, for stable, reproducible output.
    """
    weights = severity or DEFAULT_SEVERITY
    items: list[RankedItem] = []
    for v in verdicts:
        sev = weights.get(v.primary_label, 0.0)
        items.append(
            RankedItem(
                object_id=v.object_id,
                primary_label=v.primary_label,
                flags=list(v.flags),
                confidence=v.confidence,
                severity=sev,
                expected_cost=v.confidence * sev,
                rationale=v.rationale,
                recommended_action=v.recommended_action.value,
            )
        )

    if by == "confidence":
        key = lambda it: (it.confidence, it.object_id)  # noqa: E731
    elif by == "expected_cost":
        key = lambda it: (it.expected_cost, it.confidence, it.object_id)  # noqa: E731
    else:
        raise ValueError(f"unknown ranking key: {by!r}")

    return sorted(items, key=key, reverse=True)
