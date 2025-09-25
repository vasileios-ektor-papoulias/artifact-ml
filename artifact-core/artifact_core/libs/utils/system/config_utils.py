import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class DomainToolkitConfigType(Enum):
    TABLE_COMPARISON = "table_comparison.json"
    BINARY_CLASSIFICATION = "binary_classification.json"


class ConfigOverrideLocator:
    _config_dir_name = ".artifact-ml"

    @classmethod
    def get_config_override_dir(cls) -> Optional[Path]:
        user_config_dir = cls._ascend_to_marker(marker=cls._config_dir_name)
        if user_config_dir is not None:
            return user_config_dir

    @classmethod
    def get_config_override(
        cls, domain_toolkit_config_type: DomainToolkitConfigType
    ) -> Optional[Dict[str, Any]]:
        user_config_dir = cls._ascend_to_marker(marker=cls._config_dir_name)
        if user_config_dir is not None:
            user_config_file = user_config_dir / domain_toolkit_config_type.value
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


class ConfigMerger:
    @classmethod
    def merge(
        cls, base_config: Dict[str, Any], override: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if override is None:
            override = {}
        merged_config = base_config.copy()
        for key, value in override.items():
            if (
                key in merged_config
                and isinstance(merged_config[key], dict)
                and isinstance(value, dict)
            ):
                merged_config[key] = cls.merge(merged_config[key], value)
            else:
                merged_config[key] = value
        return merged_config
