from app.prompts import prompt


def test_default_when_no_override(tmp_path, monkeypatch):
    monkeypatch.setattr("app.prompts.prompts_dir", lambda: tmp_path)
    assert prompt("memory", "DEFAULT") == "DEFAULT"


def test_override_from_file(tmp_path, monkeypatch):
    monkeypatch.setattr("app.prompts.prompts_dir", lambda: tmp_path)
    (tmp_path / "memory.md").write_text("custom persona\n")
    assert prompt("memory", "DEFAULT") == "custom persona"


def test_empty_file_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr("app.prompts.prompts_dir", lambda: tmp_path)
    (tmp_path / "memory.md").write_text("  \n")
    assert prompt("memory", "DEFAULT") == "DEFAULT"
