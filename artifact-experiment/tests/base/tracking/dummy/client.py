from typing import Dict

import numpy as np
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.base.tracking.logger import ArtifactLogger
from matplotlib.figure import Figure

from tests.base.tracking.dummy.backend import DummyTrackingBackend
from tests.base.tracking.dummy.logger import (
    DummyArtifactLogger,
)


class DummyTrackingClient(TrackingClient[DummyTrackingBackend]):
    def __init__(self, run: DummyTrackingBackend):
        super().__init__(run=run)

    @staticmethod
    def _get_score_logger(
        run: DummyTrackingBackend,
    ) -> ArtifactLogger[float, DummyTrackingBackend]:
        return DummyArtifactLogger(run=run)

    @staticmethod
    def _get_array_logger(
        run: DummyTrackingBackend,
    ) -> ArtifactLogger[np.ndarray, DummyTrackingBackend]:
        return DummyArtifactLogger(run=run)

    @staticmethod
    def _get_plot_logger(
        run: DummyTrackingBackend,
    ) -> ArtifactLogger[Figure, DummyTrackingBackend]:
        return DummyArtifactLogger(run=run)

    @staticmethod
    def _get_score_collection_logger(
        run: DummyTrackingBackend,
    ) -> ArtifactLogger[Dict[str, float], DummyTrackingBackend]:
        return DummyArtifactLogger(run=run)

    @staticmethod
    def _get_array_collection_logger(
        run: DummyTrackingBackend,
    ) -> ArtifactLogger[Dict[str, np.ndarray], DummyTrackingBackend]:
        return DummyArtifactLogger(run=run)

    @staticmethod
    def _get_plot_collection_logger(
        run: DummyTrackingBackend,
    ) -> ArtifactLogger[Dict[str, Figure], DummyTrackingBackend]:
        return DummyArtifactLogger(run=run)

    def upload(self, path_source: str, dir_target: str):
        _ = path_source
        _ = dir_target
        pass
