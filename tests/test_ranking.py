from proofreader_mcp.export import export_queue
from proofreader_mcp.models import PrimaryLabel, RecommendedAction, Verdict
from proofreader_mcp.ranking import DEFAULT_SEVERITY, expected_cost, rank


def _v(oid, label, conf):
    return Verdict(
        object_id=oid,
        primary_label=label,
        confidence=conf,
        rationale="r",
        recommended_action=RecommendedAction.NONE,
    )


def test_expected_cost_is_confidence_times_severity():
    v = _v("1", PrimaryLabel.LIKELY_OVERMERGED, 0.5)
    assert expected_cost(v) == 0.5 * DEFAULT_SEVERITY[PrimaryLabel.LIKELY_OVERMERGED]


def test_acceptable_has_zero_cost():
    v = _v("1", PrimaryLabel.ACCEPTABLE, 1.0)
    assert expected_cost(v) == 0.0


def test_rank_orders_by_expected_cost_desc():
    verdicts = [
        _v("low", PrimaryLabel.JUNK, 0.9),  # 0.9 * 0.3 = 0.27
        _v("high", PrimaryLabel.LIKELY_OVERMERGED, 0.8),  # 0.8 * 1.0 = 0.80
        _v("mid", PrimaryLabel.LIKELY_OVERSEGMENTED, 0.5),  # 0.5 * 0.8 = 0.40
    ]
    ordered = [it.object_id for it in rank(verdicts)]
    assert ordered == ["high", "mid", "low"]


def test_rank_by_confidence_ignores_severity():
    verdicts = [
        _v("a", PrimaryLabel.JUNK, 0.95),
        _v("b", PrimaryLabel.LIKELY_OVERMERGED, 0.10),
    ]
    ordered = [it.object_id for it in rank(verdicts, by="confidence")]
    assert ordered == ["a", "b"]


def test_rank_is_stable_and_reproducible():
    verdicts = [
        _v("y", PrimaryLabel.JUNK, 0.5),
        _v("x", PrimaryLabel.JUNK, 0.5),
    ]
    # identical expected_cost + confidence -> deterministic object_id tiebreak
    ordered = [it.object_id for it in rank(verdicts)]
    assert ordered == ["y", "x"]  # reverse-sorted object_id on ties


def test_export_csv_has_header_and_rows():
    items = rank([_v("1", PrimaryLabel.LIKELY_OVERMERGED, 0.8)])
    csv_text = export_queue(items, fmt="csv")
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("object_id,primary_label,flags")
    assert lines[1].startswith("1,likely_overmerged")


def test_export_json_roundtrips():
    import json

    items = rank([_v("1", PrimaryLabel.JUNK, 0.3)])
    data = json.loads(export_queue(items, fmt="json"))
    assert data[0]["object_id"] == "1"
    assert data[0]["primary_label"] == "junk"
