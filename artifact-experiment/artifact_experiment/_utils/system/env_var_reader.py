import os
from getpass import getpass
from typing import Optional

from artifact_experiment._utils.system.env_detector import EnvironmentDetector


class EnvVarReader:
    @classmethod
    def get(cls, env_var_name: str, prompt: Optional[str] = None, set_env: bool = False) -> str:
        env_var = os.getenv(key=env_var_name)
        if env_var is not None:
            return env_var
        if cls._in_headed_environment():
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

    @classmethod
    def _in_headed_environment(cls) -> bool:
        return EnvironmentDetector.in_headed_environment()

    @staticmethod
    def _get_default_prompt(env_var_name: str) -> str:
        return f"{env_var_name}: "
