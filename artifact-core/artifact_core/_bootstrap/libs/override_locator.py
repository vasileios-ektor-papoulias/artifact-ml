import json
from pathlib import Path
from typing import Any, Dict, Optional

from artifact_core._bootstrap.libs.config_identifier import ToolkitConfigIdentifier
from artifact_core._bootstrap.toolkit import DomainToolkit


class ConfigOverrideLocator:
    _config_dir_name = ".artifact-ml"

    @classmethod
    def get_config_override_dir(cls) -> Optional[Path]:
        user_config_dir = cls._ascend_to_marker(marker=cls._config_dir_name)
        if user_config_dir is not None:
            return user_config_dir

    @classmethod
    def get_config_override(cls, domain_toolkit: DomainToolkit) -> Optional[Dict[str, Any]]:
        user_config_dir = cls._ascend_to_marker(marker=cls._config_dir_name)
        if user_config_dir is not None:
            config_filename = ToolkitConfigIdentifier.get_config_filename(
                domain_toolkit=domain_toolkit
            )
            user_config_file = user_config_dir / config_filename
            if user_config_file.exists():
                with user_config_file.open() as f:
                    user_overrides = json.load(f)
                return user_overrides

    @staticmethod
    def _ascend_to_marker(marker: str, start: Optional[Path] = None) -> Optional[Path]:
        if start is None:
            start = Path.cwd()
        current = start
        while current != current.parent:
            if (current / marker).exists() and (current / marker).is_dir():
                return current / marker
            current = current.parent
