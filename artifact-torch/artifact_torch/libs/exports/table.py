import os
import tempfile

import pandas as pd
from artifact_experiment.base.tracking.client import TrackingClient

from artifact_torch.libs.exports.exporter import Exporter


class TableExporter(Exporter[pd.DataFrame]):
    _tabular_data_target_dir = "tabular_data"

    @classmethod
    def _export(
        cls, data: pd.DataFrame, tracking_client: TrackingClient, target_dir: str, filename: str
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_source = os.path.join(tmpdir, filename)
            data.to_csv(path_source, index=False)
            tracking_client.upload(path_source=path_source, dir_target=target_dir)

    @classmethod
    def _get_target_dir(cls) -> str:
        return cls._tabular_data_target_dir
