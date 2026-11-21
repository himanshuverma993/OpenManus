import multiprocessing
import sys
from io import StringIO
from typing import Dict

from app.tool.base import BaseTool


class PythonExecute(BaseTool):
    """A tool for executing Python code with timeout and safety restrictions."""

    name: str = "python_execute"
    description: str = "Executes Python code string. Note: Only print outputs are visible, function return values are not captured. Use print statements to see results."
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
        },
        "required": ["code"],
    }

    def _run_code(self, code: str, result_dict: dict, safe_globals: dict) -> None:
        original_stdout = sys.stdout
        try:
            output_buffer = StringIO()
            sys.stdout = output_buffer
            exec(code, safe_globals, safe_globals)
            result_dict["observation"] = output_buffer.getvalue()
            result_dict["success"] = True
        except Exception as e:
            result_dict["observation"] = str(e)
            result_dict["success"] = False
        finally:
            sys.stdout = original_stdout

    async def execute(
        self,
        code: str,
        timeout: int = 5,
    ) -> Dict:
        """
        Executes the provided Python code with a timeout.

        Args:
            code (str): The Python code to execute.
            timeout (int): Execution timeout in seconds.

        Returns:
            Dict: Contains 'output' with execution output or error message and 'success' status.
        """

        try:
            with multiprocessing.Manager() as manager:
                result = manager.dict({"observation": "", "success": False})
                if isinstance(__builtins__, dict):
                    safe_globals = {"__builtins__": __builtins__}
                else:
                    safe_globals = {"__builtins__": __builtins__.__dict__.copy()}
                proc = multiprocessing.Process(
                    target=self._run_code, args=(code, result, safe_globals)
                )
                proc.start()
                proc.join(timeout)

                # timeout process
                if proc.is_alive():
                    proc.terminate()
                    proc.join(1)
                    return {
                        "observation": f"Execution timeout after {timeout} seconds",
                        "success": False,
                    }
                return dict(result)
        except Exception as exc:
            # `multiprocessing` can be unavailable or restricted inside notebook
            # kernels (e.g. when the start method is "spawn" and the notebook
            # cannot be re-imported). Fall back to running the code inline so the
            # agent keeps working, at the cost of losing the hard timeout.
            result: Dict = {}
            self._run_code(code, result, self._safe_globals())
            result.setdefault("observation", "")
            result.setdefault("success", False)
            result["observation"] = (
                f"{result['observation']}\n"
                f"[note] executed without a timeout because multiprocessing was "
                f"unavailable ({type(exc).__name__}: {exc})"
            )
            return result

    @staticmethod
    def _safe_globals() -> Dict:
        if isinstance(__builtins__, dict):
            return {"__builtins__": __builtins__}
        return {"__builtins__": __builtins__.__dict__.copy()}
