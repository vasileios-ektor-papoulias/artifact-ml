from typing import Dict, Optional, Type, TypeVar

from matplotlib.figure import Figure
from mlflow.tracking import MlflowClient
from numpy import ndarray

from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.libs.tracking.mlflow.adapter import MlflowNativeClient, MlflowRunAdapter
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

mflowTrackingClientT = TypeVar("mflowTrackingClientT", bound="MlflowTrackingClient")


class MlflowTrackingClient(TrackingClient[MlflowRunAdapter]):
    @classmethod
    def build(
        cls: Type[mflowTrackingClientT], experiment_id: str, run_id: Optional[str] = None
    ) -> mflowTrackingClientT:
        run = MlflowRunAdapter.build(experiment_id=experiment_id, run_id=run_id)
        client = cls(run=run)
        return client

    @classmethod
    def from_native_run(
        cls: Type[mflowTrackingClientT], native_run: MlflowNativeClient
    ) -> mflowTrackingClientT:
        run = MlflowRunAdapter.from_native_run(native_run=native_run)
        client = cls(run=run)
        return client

    @classmethod
    def create_experiment(cls, experiment_name: str) -> str:
        native_client = MlflowClient(tracking_uri=MlflowRunAdapter.TRACKING_URI)
        experiment = native_client.get_experiment_by_name(name=experiment_name)
        if experiment is None:
            experiment_id = native_client.create_experiment(name=experiment_name)
        else:
            experiment_id = experiment.experiment_id
        return experiment_id

    @staticmethod
    def _get_score_logger(run: MlflowRunAdapter) -> ArtifactLogger[float, MlflowRunAdapter]:
        return MlflowScoreLogger(run=run)

    @staticmethod
    def _get_array_logger(
        run: MlflowRunAdapter,
    ) -> ArtifactLogger[ndarray, MlflowRunAdapter]:
        return MlflowArrayLogger(run=run)

    @staticmethod
    def _get_plot_logger(run: MlflowRunAdapter) -> ArtifactLogger[Figure, MlflowRunAdapter]:
        return MLFlowPlotLogger(run=run)

    @staticmethod
    def _get_score_collection_logger(
        run: MlflowRunAdapter,
    ) -> ArtifactLogger[Dict[str, float], MlflowRunAdapter]:
        return MlflowScoreCollectionLogger(run=run)

    @staticmethod
    def _get_array_collection_logger(
        run: MlflowRunAdapter,
    ) -> ArtifactLogger[Dict[str, ndarray], MlflowRunAdapter]:
        return MlflowArrayCollectionLogger(run=run)

    @staticmethod
    def _get_plot_collection_logger(
        run: MlflowRunAdapter,
    ) -> ArtifactLogger[Dict[str, Figure], MlflowRunAdapter]:
        return MLFlowPlotCollectionLogger(run=run)
