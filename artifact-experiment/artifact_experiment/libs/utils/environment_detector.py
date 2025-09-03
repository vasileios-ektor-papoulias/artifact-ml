import sys
from enum import Enum
from typing import Optional


class Environment(Enum):
    TERMINAL = "TERMINAL"
    IPYTHON_TERMINAL = "IPYTHON_TERMINAL"
    JUPYTER_NOTEBOOK = "JUPYTER_NOTEBOOK"


class EnvironmentDetector:
    _ls_headed_environments = [
        Environment.TERMINAL,
        Environment.IPYTHON_TERMINAL,
        Environment.JUPYTER_NOTEBOOK,
    ]

    @classmethod
    def detect(cls) -> Environment:
        if cls._in_terminal():
            return Environment.TERMINAL
        elif cls._in_ipython_terminal():
            return Environment.IPYTHON_TERMINAL
        elif cls._in_jupyter_notebook():
            return Environment.JUPYTER_NOTEBOOK
        else:
            raise RuntimeError(
                "Unable to detect a headed environment. Possibly running headlessly."
            )

    @classmethod
    def in_headed_environment(cls) -> bool:
        try:
            return cls.detect() in cls._ls_headed_environments
        except RuntimeError:
            return False

    @staticmethod
    def _in_terminal() -> bool:
        return sys.stdin.isatty()

    @classmethod
    def _in_ipython_terminal(cls) -> bool:
        ipython_shell = cls._get_ipython_shell()
        if ipython_shell is not None:
            return cls._is_terminal_interactive_shell(obj=ipython_shell)
        else:
            return False

    @staticmethod
    def _is_terminal_interactive_shell(obj: object) -> bool:
        obj_type = type(obj)
        return (
            obj_type.__name__ == "TerminalInteractiveShell"
            and "ipython.terminal" in obj_type.__module__.lower()
        )

    @classmethod
    def _in_jupyter_notebook(cls) -> bool:
        ipython_shell = cls._get_ipython_shell()
        if ipython_shell is not None:
            return cls._is_zqm_interactive_shell(obj=ipython_shell)
        else:
            return False

    @staticmethod
    def _is_zqm_interactive_shell(obj: object) -> bool:
        obj_type = type(obj)
        return (
            obj_type.__name__ == "ZMQInteractiveShell"
            and "ipykernel" in obj_type.__module__.lower()
        )

    @staticmethod
    def _get_ipython_shell() -> Optional[object]:
        try:
            from IPython.core.getipython import get_ipython

            return get_ipython()
        except ImportError:
            pass
