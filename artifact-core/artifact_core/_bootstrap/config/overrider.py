import json
from pathlib import Path
from typing import Any, Dict, Optional

from artifact_core._bootstrap.config.filename_resolver import ConfigFilenameResolver
from artifact_core._bootstrap.types.toolkit import DomainToolkit
from artifact_core._bootstrap.utils.directory_locator import DirectoryLocator


class ConfigOverrider:
    _config_dir_name = ".artifact-ml"

    @classmethod
    def get_config_override_dir(cls) -> Optional[Path]:
        user_config_dir = DirectoryLocator.find(marker=cls._config_dir_name)
        if user_config_dir is not None:
            return user_config_dir

    @classmethod
    def get_config_override(cls, domain_toolkit: DomainToolkit) -> Optional[Dict[str, Any]]:
        user_config_dir = DirectoryLocator.find(marker=cls._config_dir_name)
        if user_config_dir is not None:
            config_filename = ConfigFilenameResolver.get_config_filename(
                domain_toolkit=domain_toolkit
            )
            user_config_file = user_config_dir / config_filename
            if user_config_file.exists():
                with user_config_file.open() as f:
                    user_overrides = json.load(f)
                return user_overrides
