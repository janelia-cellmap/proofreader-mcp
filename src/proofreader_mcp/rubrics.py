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
  networks with MANY THIN BRANCHES — thin/branchy is normal mito topology, not
  by itself an error. Smooth surface at mesh resolution; plausible size.
- Misclassified: does not look like a mitochondrion. You do not need to name
  what it actually is — "this is not a mito" is enough. Common cues: flat
  sheet-like (more like ER / membrane), spherical and compact with no
  elongation, or any shape/texture that generally does not read as a mito.
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
This is volume EM data. Expect imaging noise, wavy/streaky line artifacts, and
contrast variation across sections. Do NOT mistake such artifacts for real
structure or for a fault — judge topology from the mesh and only treat features
that persist across sections as real.

You see only a few EM cross-sections per axis, so the evidence is partial: a
membrane that separates two parts, or the natural end of a branch, may simply
fall between the sections you were given. Weigh that before calling a fault.

Each EM section is a 2D plane cutting through a 3D object. A single connected
object routinely appears as TWO OR MORE SEPARATE PROFILES in one slice — a U-,
C-, or branched shape is cut by the plane in multiple places, so disconnected
blobs in one section do NOT mean two objects. Use the MESH panels (which show
the true 3D shape) plus the fact that all profiles carry the SAME object tint to
reason about connectivity. Reconcile across X/Y/Z: profiles that look separate in
one axis are usually obviously one connected piece in another. NEVER call a merge
from "separate profiles in section N" alone — that is expected slicing geometry,
not evidence.

Structural faults (apply to every object, no rubric required):
- likely_overmerged (false merge): two genuinely separate structures joined in
  one label. The tell is a DISTINCT MEMBRANE running between two parts that are
  connected in the 3D MESH — i.e. the mesh shows them joined, but a membrane in
  the EM says they should be separate objects. That membrane may or may not be
  captured in the few slices you have; if you cannot see a clear dividing
  membrane, do not assert a merge. Also covered: implausible topology, multiple
  cell bodies in one label. NOTE: a thin branch connecting two lobes is NOT a
  merge — thin branches are normal mito topology. And separate 2D profiles within
  a slice are NOT a merge (see slicing-geometry note above) — judge connectivity
  from the mesh, not from a single section.
  CRITICAL BAR FOR CALLING A MERGE: a membrane between two lobes IN A GIVEN PLANE
  does NOT prove they are separate — they can be a single object connected
  somewhere else (e.g. a U whose two arms touch with a membrane between them in
  this slice but join around the bend, out of these sections). To call
  overmerged you must be confident the visible evidence rules OUT any connection
  anywhere — i.e. the parts are separated everywhere the structure exists, not
  just where this slice happens to cut. Because the slices are few and sporadic,
  you usually CANNOT establish that; when a connection elsewhere is plausible,
  use uncertain (human_review), not likely_overmerged.
- likely_oversegmented (false split): the object does not look like a complete,
  correct mito shape — e.g. a harsh planar truncation face mid-structure where it
  is cut flat instead of ending naturally, or a jagged/torn boundary that breaks
  the expected smooth tubular form. TRAP: the same flat cut occurs when an object
  is legitimately clipped by the dataset / field-of-view boundary. A cut AWAY
  FROM any volume edge is suspicious; a cut AT the edge is expected. Use
  meta.at_volume_edge when present to disambiguate.

When the evidence is genuinely inconclusive — e.g. a very thin or unlikely
branch where the visible sections do not show whether a dividing membrane is
present — do NOT guess. Use primary_label=uncertain with
recommended_action=human_review, and say in the rationale what additional
evidence (which sections / a finer crop) would settle it.
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
