import json
from pathlib import Path

import pytest

from proofreader_mcp.models import (
    PrimaryLabel,
    RecommendedAction,
    Review,
    ReviewVerdict,
    Verdict,
)
from proofreader_mcp.storage import STATE_FILENAME, CollectionError, Store


def test_load_collection_from_sidecar(collection_dir: Path):
    store = Store()
    report = store.load_collection(collection_dir)
    assert report.collection_id == "test_collection"
    assert report.organelle_type == "mitochondrion"
    assert report.object_count == 3
    assert report.has_rubric is True
    assert report.missing_images == []
    assert report.orphan_files == []
    assert report.source == "sidecar"


def test_image_legend_from_sidecar(tmp_path: Path):
    (tmp_path / "1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    sidecar = {
        "collection_id": "c",
        "image_legend": "row per axis; mesh + EM sections",
        "objects": [{"object_id": "1", "image": "1.png"}],
    }
    (tmp_path / "objects.json").write_text(json.dumps(sidecar))
    store = Store()
    store.load_collection(tmp_path)
    assert store.get("c").image_legend == "row per axis; mesh + EM sections"


def test_image_legend_absent_is_none(collection_dir: Path):
    store = Store()
    store.load_collection(collection_dir)
    assert store.get("test_collection").image_legend is None


def test_load_collection_by_scan(scan_dir: Path):
    store = Store()
    report = store.load_collection(scan_dir)
    assert report.source == "scan"
    assert report.organelle_type is None
    assert report.has_rubric is False
    assert report.object_count == 2
    assert sorted(store.get(report.collection_id).objects) == ["a", "b"]


def test_missing_and_orphan_detection(tmp_path: Path):
    (tmp_path / "1.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # declared + present
    (tmp_path / "orphan.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # not declared
    sidecar = {
        "collection_id": "c",
        "objects": [
            {"object_id": "1", "image": "1.png"},
            {"object_id": "2", "image": "2.png"},  # declared, absent
        ],
    }
    (tmp_path / "objects.json").write_text(json.dumps(sidecar))

    store = Store()
    report = store.load_collection(tmp_path)
    assert report.missing_images == ["2.png"]
    assert report.orphan_files == ["orphan.png"]


def test_image_filename_fallback(tmp_path: Path):
    (tmp_path / "42.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    sidecar = {"collection_id": "c", "objects": [{"object_id": "42"}]}  # no image key
    (tmp_path / "objects.json").write_text(json.dumps(sidecar))

    store = Store()
    store.load_collection(tmp_path)
    assert store.get("c").objects["42"].image == "42.png"


def test_classify_persists_and_hashes(collection_dir: Path):
    store = Store()
    store.load_collection(collection_dir)
    v = Verdict(
        object_id="100",
        primary_label=PrimaryLabel.LIKELY_OVERMERGED,
        confidence=0.9,
        rationale="two fused structures",
        recommended_action=RecommendedAction.SPLIT,
    )
    stored = store.add_verdict("test_collection", v)
    assert stored.image_hash is not None and len(stored.image_hash) == 64

    state = json.loads((collection_dir / STATE_FILENAME).read_text())
    assert state["verdicts"]["100"]["primary_label"] == "likely_overmerged"
    assert state["verdicts"]["100"]["image_hash"] == stored.image_hash


def test_state_reloads_on_second_load(collection_dir: Path):
    store = Store()
    store.load_collection(collection_dir)
    store.add_verdict(
        "test_collection",
        Verdict(
            object_id="200",
            primary_label=PrimaryLabel.JUNK,
            confidence=0.5,
            rationale="speckle",
            recommended_action=RecommendedAction.DISCARD,
        ),
    )
    store.add_review(
        "test_collection",
        Review(object_id="200", verdict=ReviewVerdict.ACCEPT, notes="agreed"),
    )

    fresh = Store()
    fresh.load_collection(collection_dir)
    col = fresh.get("test_collection")
    assert col.verdicts["200"].primary_label is PrimaryLabel.JUNK
    assert col.reviews["200"].verdict is ReviewVerdict.ACCEPT


def test_confidence_bounds_enforced(collection_dir: Path):
    store = Store()
    store.load_collection(collection_dir)
    with pytest.raises(CollectionError):
        store.add_verdict(
            "test_collection",
            Verdict(
                object_id="100",
                primary_label=PrimaryLabel.JUNK,
                confidence=1.5,
                rationale="x",
                recommended_action=RecommendedAction.NONE,
            ),
        )


def test_classify_unknown_object_rejected(collection_dir: Path):
    store = Store()
    store.load_collection(collection_dir)
    with pytest.raises(CollectionError):
        store.add_verdict(
            "test_collection",
            Verdict(
                object_id="999",
                primary_label=PrimaryLabel.JUNK,
                confidence=0.5,
                rationale="x",
                recommended_action=RecommendedAction.NONE,
            ),
        )


def test_list_objects_pagination(collection_dir: Path):
    store = Store()
    store.load_collection(collection_dir)
    page0 = store.list_objects("test_collection", page=0, page_size=2)
    assert page0["returned"] == 2
    assert page0["has_more"] is True
    page1 = store.list_objects("test_collection", page=1, page_size=2)
    assert page1["returned"] == 1
    assert page1["has_more"] is False


def test_unknown_collection_raises():
    store = Store()
    with pytest.raises(CollectionError):
        store.get("nope")
