from typing import Any, Mapping, Optional


class MapMerger:
    @classmethod
    def merge(
        cls, base: Mapping[str, Any], override: Optional[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        if override is None:
            override = {}
        merged = dict(base)
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = cls.merge(merged[key], value)
            else:
                merged[key] = value
        return merged
