"""Per-organelle validity rubrics.

The validity classes (`junk`, `misclassified`) are judged against a short rubric
that the host injects into its prompt for the declared ``organelle_type``. The
framework is generic; mitochondrion is the first rubric. The server only serves
these strings — it performs no judgment itself.
"""

from __future__ import annotations

MITOCHONDRION_RUBRIC = """\
Mitochondrion validity rubric:
- Valid: roughly tubular or ellipsoidal; may be elongated or form branched
  networks; smooth surface at mesh resolution; plausible size.
- Misclassified: flat sheet-like (more like ER / membrane), or spherical and
  compact with no elongation (possibly another organelle).
- Junk: tiny isolated blob below plausible size, disconnected speckle, or an
  angular/boxy artifact. Confirm size via meta.voxel_count rather than by eye
  when it is available.
"""

# Keyed by organelle_type. Add new rubrics here as they are written.
RUBRICS: dict[str, str] = {
    "mitochondrion": MITOCHONDRION_RUBRIC,
}

# Generic structural guidance that always applies, regardless of organelle type.
STRUCTURAL_GUIDANCE = """\
Structural faults (apply to every object, no rubric required):
- likely_overmerged: implausible topology; two distinct structures fused;
  multiple cell bodies in one label.
- likely_oversegmented: a harsh planar truncation face mid-structure where the
  object is cut flat instead of ending naturally. TRAP: the same flat cut occurs
  when an object is legitimately clipped by the dataset / field-of-view boundary.
  A cut AWAY FROM any volume edge is suspicious; a cut AT the edge is expected.
  Use meta.at_volume_edge when present to disambiguate.
"""


def get_rubric(organelle_type: str | None) -> str:
    """Return the full guidance to inject for an organelle type.

    Always includes the structural guidance. Appends the organelle-specific
    validity rubric when one is registered; otherwise notes that validity
    classes are skipped.
    """
    parts = [STRUCTURAL_GUIDANCE]
    if organelle_type and organelle_type in RUBRICS:
        parts.append(RUBRICS[organelle_type])
    elif organelle_type:
        parts.append(
            f"No validity rubric registered for organelle_type "
            f"'{organelle_type}'; validity classes (junk, misclassified) are "
            f"skipped — apply only structural faults."
        )
    else:
        parts.append(
            "No organelle_type declared; validity classes (junk, "
            "misclassified) are skipped — apply only structural faults."
        )
    return "\n".join(parts)


def has_rubric(organelle_type: str | None) -> bool:
    return bool(organelle_type) and organelle_type in RUBRICS
