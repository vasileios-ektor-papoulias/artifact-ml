import json

from artifact_core.typing import Array


class ArrayStringifer:
    @staticmethod
    def stringify(array: Array) -> str:
        return json.dumps(array.tolist())
