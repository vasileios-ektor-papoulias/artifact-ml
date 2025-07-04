import json
import os
import tempfile
from typing import Any, Dict

from artifact_experiment.base.tracking.client import TrackingClient

from artifact_torch.libs.exports.exporter import Exporter


class MetadataExporter(Exporter[Dict[str, Any]]):
    _metadata_target_dir = "metadata"

    @classmethod
    def _export(
        cls, data: Dict[str, Any], tracking_client: TrackingClient, target_dir: str, filename: str
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_source = os.path.join(tmpdir, filename)
            with open(path_source, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            tracking_client.upload(path_source=path_source, dir_target=target_dir)

    @classmethod
    def _get_target_dir(cls) -> str:
        return cls._metadata_target_dir
