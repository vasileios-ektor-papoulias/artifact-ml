import json
import os
from abc import abstractmethod
from typing import Any, Dict

from artifact_torch.base.components.callbacks.export import ExportCallback


class MetadataExportCallback(ExportCallback[Dict[str, Any]]):
    @classmethod
    @abstractmethod
    def _get_metadata_name(cls) -> str: ...

    @classmethod
    def _get_export_name(cls) -> str:
        return cls._get_metadata_name()

    @classmethod
    def _get_extension(cls) -> str:
        return "json"

    @classmethod
    def _save_local(cls, data: Dict[str, Any], dir_target: str, filename: str) -> str:
        filepath = os.path.join(dir_target, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return filepath
