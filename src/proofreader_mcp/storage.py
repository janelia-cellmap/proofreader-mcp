"""Filesystem-backed collection registry and verdict/review state.

A *collection* is a directory of per-object composite PNGs, optionally described
by a sidecar ``objects.json``. State produced during proofreading (model verdicts
and human reviews) is persisted to ``proofreader_state.json`` inside the
collection directory, keyed by object id, so re-runs are comparable and resumable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import ObjectRecord, Review, Verdict
from .rubrics import get_rubric, has_rubric

STATE_FILENAME = "proofreader_state.json"
SIDECAR_FILENAME = "objects.json"


class CollectionError(Exception):
    """Raised for unknown collections/objects or malformed inputs."""


def hash_image(path: Path) -> str:
    """sha256 of the image bytes, used to make verdicts reproducible."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class LoadReport:
    collection_id: str
    path: str
    organelle_type: str | None
    object_count: int
    has_rubric: bool
    missing_images: list[str] = field(default_factory=list)  # declared, file absent
    orphan_files: list[str] = field(default_factory=list)  # png on disk, not declared
    source: str = "sidecar"  # "sidecar" or "scan"

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_id": self.collection_id,
            "path": self.path,
            "organelle_type": self.organelle_type,
            "object_count": self.object_count,
            "has_rubric": self.has_rubric,
            "missing_images": self.missing_images,
            "orphan_files": self.orphan_files,
            "source": self.source,
        }


@dataclass
class Collection:
    collection_id: str
    path: Path
    organelle_type: str | None
    objects: dict[str, ObjectRecord]
    verdicts: dict[str, Verdict] = field(default_factory=dict)
    reviews: dict[str, Review] = field(default_factory=dict)

    def image_path(self, object_id: str) -> Path:
        obj = self.objects.get(object_id)
        if obj is None:
            raise CollectionError(
                f"object '{object_id}' not in collection '{self.collection_id}'"
            )
        return self.path / obj.image


def _read_sidecar(directory: Path) -> dict[str, Any] | None:
    sidecar = directory / SIDECAR_FILENAME
    if not sidecar.exists():
        return None
    with open(sidecar) as fh:
        return json.load(fh)


def _state_path(directory: Path) -> Path:
    return directory / STATE_FILENAME


def _load_state(directory: Path) -> tuple[dict[str, Verdict], dict[str, Review]]:
    path = _state_path(directory)
    if not path.exists():
        return {}, {}
    with open(path) as fh:
        data = json.load(fh)
    verdicts = {
        oid: Verdict.from_dict(v) for oid, v in (data.get("verdicts") or {}).items()
    }
    reviews = {
        oid: Review.from_dict(r) for oid, r in (data.get("reviews") or {}).items()
    }
    return verdicts, reviews


class Store:
    """In-memory registry of loaded collections with persistent per-collection state."""

    def __init__(self) -> None:
        self._collections: dict[str, Collection] = {}

    # -- registration -----------------------------------------------------

    def load_collection(self, path: str | Path) -> LoadReport:
        directory = Path(path).expanduser().resolve()
        if not directory.is_dir():
            raise CollectionError(f"not a directory: {directory}")

        sidecar = _read_sidecar(directory)
        if sidecar is not None:
            collection_id = str(sidecar.get("collection_id") or directory.name)
            organelle_type = sidecar.get("organelle_type")
            objects: dict[str, ObjectRecord] = {}
            for entry in sidecar.get("objects") or []:
                rec = ObjectRecord.from_dict(
                    {
                        "object_id": entry["object_id"],
                        "image": entry.get("image")
                        or f"{entry['object_id']}.png",
                        "meta": entry.get("meta") or {},
                    }
                )
                objects[rec.object_id] = rec
            source = "sidecar"
        else:
            collection_id = directory.name
            organelle_type = None
            objects = {
                p.stem: ObjectRecord(object_id=p.stem, image=p.name)
                for p in sorted(directory.glob("*.png"))
            }
            source = "scan"

        # Validation: declared-but-absent images, and on-disk pngs not declared.
        declared_images = {rec.image for rec in objects.values()}
        missing_images = sorted(
            rec.image for rec in objects.values() if not (directory / rec.image).exists()
        )
        on_disk = {p.name for p in directory.glob("*.png")}
        orphan_files = sorted(on_disk - declared_images)

        verdicts, reviews = _load_state(directory)
        # Keep only state that still maps to a known object.
        verdicts = {oid: v for oid, v in verdicts.items() if oid in objects}
        reviews = {oid: r for oid, r in reviews.items() if oid in objects}

        collection = Collection(
            collection_id=collection_id,
            path=directory,
            organelle_type=organelle_type,
            objects=objects,
            verdicts=verdicts,
            reviews=reviews,
        )
        self._collections[collection_id] = collection

        return LoadReport(
            collection_id=collection_id,
            path=str(directory),
            organelle_type=organelle_type,
            object_count=len(objects),
            has_rubric=has_rubric(organelle_type),
            missing_images=missing_images,
            orphan_files=orphan_files,
            source=source,
        )

    # -- accessors --------------------------------------------------------

    def get(self, collection_id: str) -> Collection:
        col = self._collections.get(collection_id)
        if col is None:
            raise CollectionError(f"collection '{collection_id}' not loaded")
        return col

    def list_collection_ids(self) -> list[str]:
        return sorted(self._collections)

    def list_objects(
        self, collection_id: str, page: int = 0, page_size: int = 100
    ) -> dict[str, Any]:
        col = self.get(collection_id)
        if page < 0 or page_size <= 0:
            raise CollectionError("page must be >= 0 and page_size > 0")
        ordered = sorted(col.objects.values(), key=lambda o: o.object_id)
        start = page * page_size
        chunk = ordered[start : start + page_size]
        items = []
        for rec in chunk:
            verdict = col.verdicts.get(rec.object_id)
            review = col.reviews.get(rec.object_id)
            items.append(
                {
                    "object_id": rec.object_id,
                    "image": rec.image,
                    "meta": rec.meta,
                    "classified": verdict is not None,
                    "primary_label": verdict.primary_label.value if verdict else None,
                    "reviewed": review is not None,
                }
            )
        return {
            "collection_id": collection_id,
            "page": page,
            "page_size": page_size,
            "total": len(ordered),
            "returned": len(items),
            "has_more": start + page_size < len(ordered),
            "objects": items,
        }

    def rubric(self, collection_id: str) -> str:
        col = self.get(collection_id)
        return get_rubric(col.organelle_type)

    # -- mutation + persistence ------------------------------------------

    def add_verdict(self, collection_id: str, verdict: Verdict) -> Verdict:
        col = self.get(collection_id)
        if verdict.object_id not in col.objects:
            raise CollectionError(
                f"object '{verdict.object_id}' not in collection '{collection_id}'"
            )
        if not 0.0 <= verdict.confidence <= 1.0:
            raise CollectionError("confidence must be in [0, 1]")
        if verdict.image_hash is None:
            img = col.image_path(verdict.object_id)
            if img.exists():
                verdict.image_hash = hash_image(img)
        col.verdicts[verdict.object_id] = verdict
        self._persist(col)
        return verdict

    def add_review(self, collection_id: str, review: Review) -> Review:
        col = self.get(collection_id)
        if review.object_id not in col.objects:
            raise CollectionError(
                f"object '{review.object_id}' not in collection '{collection_id}'"
            )
        col.reviews[review.object_id] = review
        self._persist(col)
        return review

    def _persist(self, col: Collection) -> None:
        data = {
            "collection_id": col.collection_id,
            "organelle_type": col.organelle_type,
            "verdicts": {oid: v.to_dict() for oid, v in col.verdicts.items()},
            "reviews": {oid: r.to_dict() for oid, r in col.reviews.items()},
        }
        tmp = _state_path(col.path).with_suffix(".json.tmp")
        with open(tmp, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        tmp.replace(_state_path(col.path))
