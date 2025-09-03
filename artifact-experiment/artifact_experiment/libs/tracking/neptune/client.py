from typing import Dict, Optional, Type, TypeVar

import neptune
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.libs.tracking.neptune.adapter import NeptuneRunAdapter
from artifact_experiment.libs.tracking.neptune.loggers.array_collections import (
    NeptuneArrayCollectionLogger,
)
from artifact_experiment.libs.tracking.neptune.loggers.arrays import NeptuneArrayLogger
from artifact_experiment.libs.tracking.neptune.loggers.plot_collections import (
    NeptunePlotCollectionLogger,
)
from artifact_experiment.libs.tracking.neptune.loggers.plots import NeptunePlotLogger
from artifact_experiment.libs.tracking.neptune.loggers.score_collections import (
    NeptuneScoreCollectionLogger,
)
from artifact_experiment.libs.tracking.neptune.loggers.scores import NeptuneScoreLogger

NeptuneTrackingClientT = TypeVar("NeptuneTrackingClientT", bound="NeptuneTrackingClient")


class NeptuneTrackingClient(TrackingClient[NeptuneRunAdapter]):
    @classmethod
    def build(
        cls: Type[NeptuneTrackingClientT], experiment_id: str, run_id: Optional[str] = None
    ) -> NeptuneTrackingClientT:
        run = NeptuneRunAdapter.build(experiment_id=experiment_id, run_id=run_id)
        client = cls._build(run=run)
        return client

    @classmethod
    def from_native_run(
        cls: Type[NeptuneTrackingClientT], native_run: neptune.Run
    ) -> NeptuneTrackingClientT:
        run = NeptuneRunAdapter.from_native_run(native_run=native_run)
        client = cls._build(run=run)
        return client

    @staticmethod
    def _get_score_logger(run: NeptuneRunAdapter) -> ArtifactLogger[float, NeptuneRunAdapter]:
        return NeptuneScoreLogger(run=run)

    @staticmethod
    def _get_array_logger(
        run: NeptuneRunAdapter,
    ) -> ArtifactLogger[ndarray, NeptuneRunAdapter]:
        return NeptuneArrayLogger(run=run)

    @staticmethod
    def _get_plot_logger(run: NeptuneRunAdapter) -> ArtifactLogger[Figure, NeptuneRunAdapter]:
        return NeptunePlotLogger(run=run)

    @staticmethod
    def _get_score_collection_logger(
        run: NeptuneRunAdapter,
    ) -> ArtifactLogger[Dict[str, float], NeptuneRunAdapter]:
        return NeptuneScoreCollectionLogger(run=run)

    @staticmethod
    def _get_array_collection_logger(
        run: NeptuneRunAdapter,
    ) -> ArtifactLogger[Dict[str, ndarray], NeptuneRunAdapter]:
        return NeptuneArrayCollectionLogger(run=run)

    @staticmethod
    def _get_plot_collection_logger(
        run: NeptuneRunAdapter,
    ) -> ArtifactLogger[Dict[str, Figure], NeptuneRunAdapter]:
        return NeptunePlotCollectionLogger(run=run)
