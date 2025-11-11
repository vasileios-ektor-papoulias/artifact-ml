from typing import Any, Dict, Optional


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
