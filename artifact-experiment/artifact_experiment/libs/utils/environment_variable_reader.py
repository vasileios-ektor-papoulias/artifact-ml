import os
import sys
from getpass import getpass
from typing import Optional


class EnvironmentVariableReader:
    def __init__(self, env_var_name: str, prompt: Optional[str] = None):
        if prompt is None:
            prompt = self._get_default_prompt(env_var_name=env_var_name)
        self.env_var_name = env_var_name
        self.prompt = prompt

    def get(self) -> str:
        env_var = os.getenv(self.env_var_name)
        if env_var is not None:
            value = env_var
        elif self._in_interactive_environment():
            value = getpass(self.prompt)
        else:
            raise RuntimeError(
                f"{self.env_var_name} must be set as an environment variable "
                f"in non-interactive environments."
            )
        return value

    @staticmethod
    def _in_interactive_environment() -> bool:
        try:
            from IPython.core.getipython import get_ipython

            if get_ipython():
                return True
        except ImportError:
            pass
        return sys.stdin.isatty()

    @staticmethod
    def _get_default_prompt(env_var_name: str) -> str:
        return f"Enter value for {env_var_name}:"
