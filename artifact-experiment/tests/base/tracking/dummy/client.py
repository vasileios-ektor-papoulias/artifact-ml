from typing import Dict, Optional, Type, TypeVar

import numpy as np
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.base.tracking.logger import ArtifactLogger
from matplotlib.figure import Figure

from tests.base.tracking.dummy.adapter import DummyNativeRun, DummyRunAdapter
from tests.base.tracking.dummy.logger import (
    DummyArtifactLogger,
)

DummyTrackingClientT = TypeVar("DummyTrackingClientT", bound="DummyTrackingClient")


class DummyTrackingClient(TrackingClient[DummyRunAdapter]):
    @classmethod
    def build(
        cls: Type[DummyTrackingClientT], experiment_id: str, run_id: Optional[str] = None
    ) -> DummyTrackingClientT:
        run = DummyRunAdapter.build(experiment_id=experiment_id, run_id=run_id)
        client = cls._build(run=run)
        return client

    @classmethod
    def from_native_run(
        cls: Type[DummyTrackingClientT], native_run: DummyNativeRun
    ) -> DummyTrackingClientT:
        run = DummyRunAdapter.from_native_run(native_run=native_run)
        client = cls._build(run=run)
        return client

    @staticmethod
    def _get_score_logger(
        run: DummyRunAdapter,
    ) -> ArtifactLogger[float, DummyRunAdapter]:
        return DummyArtifactLogger(run=run)

    @staticmethod
    def _get_array_logger(
        run: DummyRunAdapter,
    ) -> ArtifactLogger[np.ndarray, DummyRunAdapter]:
        return DummyArtifactLogger(run=run)

    @staticmethod
    def _get_plot_logger(
        run: DummyRunAdapter,
    ) -> ArtifactLogger[Figure, DummyRunAdapter]:
        return DummyArtifactLogger(run=run)

    @staticmethod
    def _get_score_collection_logger(
        run: DummyRunAdapter,
    ) -> ArtifactLogger[Dict[str, float], DummyRunAdapter]:
        return DummyArtifactLogger(run=run)

    @staticmethod
    def _get_array_collection_logger(
        run: DummyRunAdapter,
    ) -> ArtifactLogger[Dict[str, np.ndarray], DummyRunAdapter]:
        return DummyArtifactLogger(run=run)

    @staticmethod
    def _get_plot_collection_logger(
        run: DummyRunAdapter,
    ) -> ArtifactLogger[Dict[str, Figure], DummyRunAdapter]:
        return DummyArtifactLogger(run=run)

    def upload(self, path_source: str, dir_target: str):
        self._run.upload(path_source=path_source, dir_target=dir_target)
