import os

import pandas as pd

from artifact_torch.base.components.callbacks.export import ExportCallback, ExportCallbackResources

TableExportCallbackResources = ExportCallbackResources[pd.DataFrame]


class TableExportCallback(ExportCallback[pd.DataFrame]):
    _export_name = "TABULAR_DATA"

    @classmethod
    def _get_export_name(cls) -> str:
        return cls._export_name

    @classmethod
    def _get_extension(cls) -> str:
        return "csv"

    @classmethod
    def _save_local(cls, data: pd.DataFrame, dir_target: str, filename: str) -> str:
        filepath = os.path.join(dir_target, filename)
        data.to_csv(filepath, index=False)
        return filepath
