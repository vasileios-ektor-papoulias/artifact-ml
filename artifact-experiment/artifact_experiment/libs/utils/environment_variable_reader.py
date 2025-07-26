import os
import sys
from getpass import getpass
from typing import Optional


class EnvironmentVariableReader:
    @classmethod
    def get(cls, env_var_name: str, prompt: Optional[str] = None, set_env: bool = False) -> str:
        env_var = os.getenv(key=env_var_name)
        if env_var is not None:
            return env_var

        if cls._in_interactive_environment():
            if prompt is None:
                prompt = cls._get_default_prompt(env_var_name)
            return cls._get_from_prompt(env_var_name, prompt=prompt, set_env=set_env)

        raise RuntimeError(
            f"{env_var_name} must be set as an environment variable "
            f"in non-interactive environments."
        )

    @classmethod
    def _get_from_prompt(cls, env_var_name: str, prompt: str, set_env: bool) -> str:
        value = getpass(prompt=prompt)
        if set_env:
            os.environ[env_var_name] = value
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
        return f"Enter {env_var_name}:"
