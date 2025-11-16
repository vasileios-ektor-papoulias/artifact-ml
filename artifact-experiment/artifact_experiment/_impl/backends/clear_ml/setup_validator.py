import os
from pathlib import Path


class ClearMLSetupValidator:
    _config_path = Path.home() / "clearml.conf"
    _host_env_var = "CLEARML_API_HOST"
    _access_key_env_var = "CLEARML_API_ACCESS_KEY"
    _secret_key_env_var = "CLEARML_API_SECRET_KEY"

    @classmethod
    def is_configured(cls) -> bool:
        has_config_file = cls._config_path.is_file()
        has_host = os.getenv(key=cls._host_env_var)
        has_access_key = os.getenv(key=cls._access_key_env_var)
        has_secret_key = os.getenv(key=cls._secret_key_env_var)
        has_api_credentials = has_access_key and has_secret_key
        if has_api_credentials and has_host:
            return True
        if has_host and not has_api_credentials:
            return True
        if has_config_file:
            return True
        return False

    @classmethod
    def get_setup_instructions(cls) -> str:
        return (
            "ClearML is not configured.\n"
            "To set it up, either:\n"
            "- Run `clearml-init` in the terminal, or\n"
            "- Set the required environment variables manually:\n"
            f"  • remote: {cls._host_env_var}, "
            f"{cls._access_key_env_var}, "
            f"{cls._secret_key_env_var}\n"
            f"  • local: {cls._host_env_var} only\n"
        )
