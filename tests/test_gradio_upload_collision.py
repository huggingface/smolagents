"""Uploads sharing an original filename must not overwrite each other."""

import os
from pathlib import Path
from unittest.mock import Mock

from smolagents.gradio_ui import GradioUI


def test_duplicate_upload_names_get_unique_paths(tmp_path: Path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    first = tmp_path / "folder1" / "report.txt"
    second = tmp_path / "folder2" / "report.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("FIRST FILE")
    second.write_text("SECOND FILE")

    ui = object.__new__(GradioUI)
    ui.file_upload_folder = str(upload_dir)
    first_saved = ui._save_uploaded_file(str(first))
    second_saved = ui._save_uploaded_file(str(second))

    assert first_saved != second_saved
    assert os.path.basename(first_saved) == "report.txt"
    assert os.path.basename(second_saved) == "report_1.txt"
    assert Path(first_saved).read_text() == "FIRST FILE"
    assert Path(second_saved).read_text() == "SECOND FILE"
