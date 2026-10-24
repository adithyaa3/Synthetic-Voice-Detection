"""G15: the numbers in README.md must equal what the saved artifacts say."""
from voicedetect.config import ROOT

from .conftest import requires_model


@requires_model
def test_readme_results_block_matches_artifacts():
    from voicedetect.readme import data_markdown, readme_block, readme_data_block, results_markdown
    assert readme_block() == results_markdown(), (
        "README results are stale or hand-edited; run: python -m voicedetect.pipeline readme")
    assert readme_data_block() == data_markdown(), "README data table is stale; run the 'readme' step"


@requires_model
def test_metadata_threshold_matches_calibration_artifact():
    import json
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    cal = json.loads((ROOT / "reports" / "selection" / "calibration.json").read_text())
    assert meta["threshold"] == cal["decision"]["threshold"]
    assert meta["calibration"] == cal["chosen_method"]
