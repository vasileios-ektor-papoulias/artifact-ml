from typing import Dict, Optional, Type

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.libs.tracking.mlflow.backend import MlflowBackend, MlflowNativeClient
from artifact_experiment.libs.tracking.mlflow.loggers.array_collections import (
    MlflowArrayCollectionLogger,
)
from artifact_experiment.libs.tracking.mlflow.loggers.arrays import MlflowArrayLogger
from artifact_experiment.libs.tracking.mlflow.loggers.plot_collections import (
    MLFlowPlotCollectionLogger,
)
from artifact_experiment.libs.tracking.mlflow.loggers.plots import MLFlowPlotLogger
from artifact_experiment.libs.tracking.mlflow.loggers.score_collections import (
    MlflowScoreCollectionLogger,
)
from artifact_experiment.libs.tracking.mlflow.loggers.scores import MlflowScoreLogger


class MlflowTrackingClient(TrackingClient[MlflowBackend]):
    @classmethod
    def build(
        cls: Type["MlflowTrackingClient"], experiment_id: str, run_id: Optional[str] = None
    ) -> "MlflowTrackingClient":
        backend = MlflowBackend.build(experiment_id=experiment_id, run_id=run_id)
        client = MlflowTrackingClient(backend=backend)
        return client

    @classmethod
    def from_native_client(
        cls: Type["MlflowTrackingClient"], native_client: MlflowNativeClient
    ) -> "MlflowTrackingClient":
        backend = MlflowBackend.from_native_client(native_client=native_client)
        client = MlflowTrackingClient(backend=backend)
        return client

    @staticmethod
    def _get_score_logger(backend: MlflowBackend) -> ArtifactLogger[float, MlflowBackend]:
        return MlflowScoreLogger(backend=backend)

    @staticmethod
    def _get_array_logger(
        backend: MlflowBackend,
    ) -> ArtifactLogger[ndarray, MlflowBackend]:
        return MlflowArrayLogger(backend=backend)

    @staticmethod
    def _get_plot_logger(backend: MlflowBackend) -> ArtifactLogger[Figure, MlflowBackend]:
        return MLFlowPlotLogger(backend=backend)

    @staticmethod
    def _get_score_collection_logger(
        backend: MlflowBackend,
    ) -> ArtifactLogger[Dict[str, float], MlflowBackend]:
        return MlflowScoreCollectionLogger(backend=backend)

    @staticmethod
    def _get_array_collection_logger(
        backend: MlflowBackend,
    ) -> ArtifactLogger[Dict[str, ndarray], MlflowBackend]:
        return MlflowArrayCollectionLogger(backend=backend)

    @staticmethod
    def _get_plot_collection_logger(
        backend: MlflowBackend,
    ) -> ArtifactLogger[Dict[str, Figure], MlflowBackend]:
        return MLFlowPlotCollectionLogger(backend=backend)
