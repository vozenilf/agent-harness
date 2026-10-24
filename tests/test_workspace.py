"""Workspace file tools - including the path traversal guard."""

import pytest

from agent_harness.workspace import Workspace


def test_list_files_shows_files_and_marks_folders(workspace, tmp_path):
    (tmp_path / "projects").mkdir()

    assert workspace.list_files() == "notes.txt\nprojects/"


def test_list_files_on_an_empty_workspace(tmp_path):
    assert Workspace(tmp_path).list_files() == "(the workspace is empty)"


def test_read_file(workspace):
    assert workspace.read_file("notes.txt") == "buy milk"


def test_write_then_read(workspace):
    assert workspace.write_file("todo.md", "- call mum") == "wrote 10 characters to todo.md"
    assert workspace.read_file("todo.md") == "- call mum"


def test_reading_a_missing_file_raises(workspace):
    with pytest.raises(FileNotFoundError):
        workspace.read_file("missing.txt")


@pytest.mark.parametrize("filename", ["../secret.txt", "../../etc/passwd", "sub/../../x"])
def test_paths_outside_the_workspace_are_refused(workspace, filename):
    with pytest.raises(PermissionError, match="outside the workspace"):
        workspace.read_file(filename)
    with pytest.raises(PermissionError, match="outside the workspace"):
        workspace.write_file(filename, "gotcha")
    assert not (workspace.root / filename).resolve().exists() 


def test_absolute_paths_are_refused(workspace, tmp_path):
    outside = tmp_path.parent / "outside.txt"

    with pytest.raises(PermissionError):
        workspace.write_file(str(outside), "gotcha")
    assert not outside.exists()


def test_tools_describe_the_three_methods(workspace):
    tools = {tool.name: tool for tool in workspace.tools()}

    assert set(tools) == {"list_files", "read_file", "write_file"}
    assert tools["read_file"].read_only
    assert not tools["write_file"].read_only  # changes files -> needs approval in phase 6