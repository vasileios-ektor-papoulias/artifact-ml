import os
import tempfile

import pandas as pd
from artifact_experiment.base.tracking.client import TrackingClient


class TabularDataExporter:
    _generations_target_dir = "genertations"

    @classmethod
    def export(
        cls,
        df: pd.DataFrame,
        tracking_client: TrackingClient,
        step: int,
    ):
        dir_target = cls._get_generation_target_dir()
        tmp_filename = cls._get_tmp_filename(step=step)
        with tempfile.TemporaryDirectory() as tmpdir:
            path_source = os.path.join(tmpdir, tmp_filename)
            df.to_csv(path_source, index=False)
            tracking_client.upload(path_source=path_source, dir_target=dir_target)

    @classmethod
    def _get_generation_target_dir(cls) -> str:
        return cls._generations_target_dir

    @classmethod
    def _get_tmp_filename(cls, step: int) -> str:
        return f"{step}.csv"
