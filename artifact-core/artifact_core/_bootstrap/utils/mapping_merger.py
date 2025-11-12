from typing import Any, Dict, Optional


class MappingMerger:
    @classmethod
    def merge(cls, base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if override is None:
            override = {}
        merged = base.copy()
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = cls.merge(merged[key], value)
            else:
                merged[key] = value
        return merged
