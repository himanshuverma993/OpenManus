import sys

from app.tool import BaseTool


class AskHuman(BaseTool):
    """Add a tool to ask human for help."""

    name: str = "ask_human"
    description: str = "Use this tool to ask human for help."
    parameters: str = {
        "type": "object",
        "properties": {
            "inquire": {
                "type": "string",
                "description": "The question you want to ask human.",
            }
        },
        "required": ["inquire"],
    }

    async def execute(self, inquire: str) -> str:
        # `input()` raises (or hangs) when stdin is not an interactive terminal,
        # which is the case inside Jupyter/Kaggle notebooks. Detect that up
        # front and hand the decision back to the agent instead of crashing.
        if not self._interactive_stdin_available():
            return (
                "Error: no interactive terminal available, so the human question "
                f"cannot be answered automatically. Proceed with the best available "
                f"information instead. Question was: {inquire}"
            )

        try:
            return input(f"""Bot: {inquire}\n\nYou: """).strip()
        except (EOFError, OSError, KeyboardInterrupt) as exc:
            return (
                "Error: could not read a human response "
                f"({type(exc).__name__}). Continue without asking a human."
            )

    @staticmethod
    def _interactive_stdin_available() -> bool:
        try:
            if sys.stdin is None or sys.stdin.closed:
                return False
            return bool(sys.stdin.isatty())
        except (AttributeError, ValueError, OSError):
            return False
