"""Local file tools, confined to one workspace folder.

The tutorial's read_file("../secret.txt") could read any file on the computer. Here
every path is resolved first and refused if it ends up outside the workspace.
"""

from pathlib import Path

from agent_harness.tools import Tool

FILENAME = {"type": "string", "description": "File name, relative to the workspace folder."}


class Workspace:
    """A folder the model may list, read and write - and nothing outside it."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def path(self, filename: str) -> Path:
        """The absolute path for a filename, refusing anything outside the workspace."""
        path = (self.root / filename).resolve()
        if not path.is_relative_to(self.root):
            raise PermissionError(f"'{filename}' is outside the workspace")
        return path

    def list_files(self) -> str:
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in self.root.iterdir())
        return "\n".join(names) or "(the workspace is empty)"

    def read_file(self, filename: str) -> str:
        return self.path(filename).read_text(encoding="utf-8")

    def write_file(self, filename: str, content: str) -> str:
        self.path(filename).write_text(content, encoding="utf-8")
        return f"wrote {len(content)} characters to {filename}"

    def tools(self) -> list[Tool]:
        """The three methods above, described for the model."""
        return [
            Tool(
                name="list_files",
                description="List the files in the user's workspace folder.",
                parameters={"type": "object", "properties": {}},
                function=self.list_files,
            ),
            Tool(
                name="read_file",
                description="Read one text file from the user's workspace folder.",
                parameters={
                    "type": "object",
                    "properties": {"filename": FILENAME},
                    "required": ["filename"],
                },
                function=self.read_file,
            ),
            Tool(
                name="write_file",
                description="Create or overwrite a text file in the user's workspace folder.",
                parameters={
                    "type": "object",
                    "properties": {
                        "filename": FILENAME,
                        "content": {"type": "string", "description": "The full new text."},
                    },
                    "required": ["filename", "content"],
                },
                function=self.write_file,
                read_only=False,
            ),
        ]
