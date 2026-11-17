"""Environment pre-flight tests: interpreter, dependencies and artefacts."""

from __future__ import annotations

from src.check_environment import (
    MINIMUM_PYTHON,
    check_files,
    check_modules,
    check_python,
)


def test_python_version_check_uses_the_documented_minimum():
    assert check_python(MINIMUM_PYTHON + (0,))["ok"] is True
    older = (MINIMUM_PYTHON[0], MINIMUM_PYTHON[1] - 1, 9)
    assert check_python(older)["ok"] is False


def test_missing_module_reports_the_import_error():
    def importer(name: str):
        if name == "missing_package":
            raise ImportError("No module named 'missing_package'")
        return object()

    results = check_modules(("pandas", "missing_package"), importer=importer)

    assert results[0] == {"name": "pandas", "ok": True, "detail": "imported"}
    assert results[1]["ok"] is False
    assert "missing_package" in results[1]["detail"]


def test_file_check_reports_missing_and_empty_files(tmp_path):
    (tmp_path / "present.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")

    results = check_files(("present.txt", "empty.txt", "absent.txt"), root=tmp_path)

    assert [item["ok"] for item in results] == [True, False, False]
    assert results[1]["detail"] == "empty file"
    assert results[2]["detail"] == "missing"
