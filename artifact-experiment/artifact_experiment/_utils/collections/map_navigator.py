from typing import Any, Mapping


class MapNavigator:
    @staticmethod
    def get(data: Mapping[str, Any], path: str, default: Any = None, separator: str = "/") -> Any:
        if not path:
            return data
        path_parts = [p for p in path.split(separator) if p]
        current = data
        for part in path_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current
