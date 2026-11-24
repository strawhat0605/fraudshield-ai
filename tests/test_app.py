from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_starts_and_explains_missing_model(tmp_path, monkeypatch):
    # Run from an isolated directory so a developer's local artifacts cannot
    # make this smoke test depend on their machine state.
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    monkeypatch.chdir(tmp_path)

    app = AppTest.from_file(str(app_path)).run()

    assert not app.exception
    assert any("FraudShield AI" in title.value for title in app.title)
    assert any("No trained model was found" in element.value for element in app.info)
