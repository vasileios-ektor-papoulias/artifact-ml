import json
from abc import abstractmethod

from artifact_experiment._base.components.callbacks.export import ExportCallback
from artifact_experiment._base.components.resources.export import ExportCallbackResources
from artifact_experiment._base.typing.metadata import Metadata


class MetadataExportCallback(ExportCallback[ExportCallbackResources[Metadata]]):
    _extension = "json"

    @classmethod
    @abstractmethod
    def _get_metadata_name(cls) -> str: ...

    @classmethod
    def _get_export_name(cls) -> str:
        return cls._get_metadata_name()

    @classmethod
    def _get_extension(cls) -> str:
        return cls._extension

    @classmethod
    def _save_local(cls, resources: ExportCallbackResources[Metadata], filepath: str) -> str:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(resources.export_data, f, indent=4)
        return filepath
