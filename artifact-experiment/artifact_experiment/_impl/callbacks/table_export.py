import pandas as pd

from artifact_experiment._base.components.callbacks.export import (
    ExportCallback,
    ExportCallbackResources,
)

TableExportCallbackResources = ExportCallbackResources[pd.DataFrame]


class TableExportCallback(ExportCallback[ExportCallbackResources[pd.DataFrame]]):
    _export_name = "TABULAR_DATA"
    _extension = "csv"

    @classmethod
    def _get_export_name(cls) -> str:
        return cls._export_name

    @classmethod
    def _get_extension(cls) -> str:
        return cls._extension

    @classmethod
    def _save_local(cls, resources: ExportCallbackResources[pd.DataFrame], filepath: str) -> str:
        resources.export_data.to_csv(filepath, index=False)
        return filepath
