"""Long-term memory: a markdown file of facts about the user.

The file is read once at startup and put into the system prompt; the model adds new
facts with the save_memory tool. Simple on purpose - a file you can open and edit.
"""

from pathlib import Path

from agent_harness.tools import Tool

MAX_FACT_CHARS = 300  # one short fact, not a document
EMPTY = "(nothing saved yet)"


class Memory:
    """Facts about the user, one per line, in a markdown file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def facts(self) -> list[str]:
        """All saved facts, without the leading "- "."""
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [line.removeprefix("- ").strip() for line in lines if line.strip()]

    def as_text(self) -> str:
        """The facts as a markdown list, for the system prompt."""
        facts = self.facts()
        return "\n".join(f"- {fact}" for fact in facts) if facts else EMPTY

    def save(self, fact: str) -> str:
        """Append one fact. Cleans it up, refuses empty or overlong ones, skips duplicates."""
        fact = " ".join(fact.split())  # one line: newlines and repeated spaces collapsed
        if not fact:
            raise ValueError("the fact is empty")
        if len(fact) > MAX_FACT_CHARS:
            raise ValueError(f"the fact is longer than {MAX_FACT_CHARS} characters")
        if fact.lower() in (known.lower() for known in self.facts()):
            return f"already known: {fact}"
        with self.path.open("a", encoding="utf-8") as file:
            file.write(f"- {fact}\n")
        return f"saved: {fact}"

    def tools(self) -> list[Tool]:
        return [
            Tool(
                name="save_memory",
                description=(
                    "Save one short, lasting fact about the user to long-term memory, "
                    "e.g. their name, preferences, projects or recurring tasks. "
                    "Do not save temporary details of the current conversation."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "fact": {"type": "string", "description": "One short sentence."}
                    },
                    "required": ["fact"],
                },
                function=self.save,
                read_only=False,
            )
        ]


def system_prompt_with_memory(base_prompt: str, memory: Memory) -> str:
    """The system prompt plus what we remember about the user."""
    return (
        f"{base_prompt}\n\n"
        f"What you remember about the user from earlier sessions:\n{memory.as_text()}\n\n"
        "When you learn a new lasting fact about the user, save it with save_memory."
    )
