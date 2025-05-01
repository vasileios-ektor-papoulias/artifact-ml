from typing import Dict, Optional, Type

import neptune
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.libs.tracking.neptune.backend import NeptuneBackend
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


class NeptuneTrackingClient(TrackingClient[NeptuneBackend]):
    @classmethod
    def build(
        cls: Type["NeptuneTrackingClient"], experiment_id: str, run_id: Optional[str] = None
    ) -> "NeptuneTrackingClient":
        backend = NeptuneBackend.build(experiment_id=experiment_id, run_id=run_id)
        client = NeptuneTrackingClient(backend=backend)
        return client

    @classmethod
    def from_native_client(
        cls: Type["NeptuneTrackingClient"], native_client: neptune.Run
    ) -> "NeptuneTrackingClient":
        backend = NeptuneBackend.from_native_client(native_client=native_client)
        client = NeptuneTrackingClient(backend=backend)
        return client

    @staticmethod
    def _get_score_logger(backend: NeptuneBackend) -> ArtifactLogger[float, NeptuneBackend]:
        return NeptuneScoreLogger(backend=backend)

    @staticmethod
    def _get_array_logger(
        backend: NeptuneBackend,
    ) -> ArtifactLogger[ndarray, NeptuneBackend]:
        return NeptuneArrayLogger(backend=backend)

    @staticmethod
    def _get_plot_logger(backend: NeptuneBackend) -> ArtifactLogger[Figure, NeptuneBackend]:
        return NeptunePlotLogger(backend=backend)

    @staticmethod
    def _get_score_collection_logger(
        backend: NeptuneBackend,
    ) -> ArtifactLogger[Dict[str, float], NeptuneBackend]:
        return NeptuneScoreCollectionLogger(backend=backend)

    @staticmethod
    def _get_array_collection_logger(
        backend: NeptuneBackend,
    ) -> ArtifactLogger[Dict[str, ndarray], NeptuneBackend]:
        return NeptuneArrayCollectionLogger(backend=backend)

    @staticmethod
    def _get_plot_collection_logger(
        backend: NeptuneBackend,
    ) -> ArtifactLogger[Dict[str, Figure], NeptuneBackend]:
        return NeptunePlotCollectionLogger(backend=backend)
