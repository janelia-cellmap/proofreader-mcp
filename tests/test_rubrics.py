from proofreader_mcp.rubrics import get_rubric, has_rubric


def test_mitochondrion_rubric_present():
    assert has_rubric("mitochondrion")
    text = get_rubric("mitochondrion")
    assert "Mitochondrion validity rubric" in text
    assert "Structural faults" in text  # structural guidance always included


def test_unknown_organelle_skips_validity():
    assert not has_rubric("golgi")
    text = get_rubric("golgi")
    assert "validity classes" in text
    assert "skipped" in text
    assert "Structural faults" in text  # structural still applies


def test_no_organelle_type():
    assert not has_rubric(None)
    text = get_rubric(None)
    assert "No organelle_type declared" in text
    assert "Structural faults" in text
