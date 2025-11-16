from artifact_core.typing import (
    Array,
    ArrayCollection,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)

from artifact_experiment._base.primitives.file import File
from artifact_experiment._base.tracking.backend.logger import ArtifactLogger, BackendLogger
from artifact_experiment._base.tracking.backend.worker import BackendLoggingWorker
from artifact_experiment._impl.backends.mlflow.adapter import MlflowRunAdapter
from artifact_experiment._impl.backends.mlflow.loggers.array_collections import (
    MlflowArrayCollectionLogger,
)
from artifact_experiment._impl.backends.mlflow.loggers.arrays import MlflowArrayLogger
from artifact_experiment._impl.backends.mlflow.loggers.files import MlflowFileLogger
from artifact_experiment._impl.backends.mlflow.loggers.plot_collections import (
    MlflowPlotCollectionLogger,
)
from artifact_experiment._impl.backends.mlflow.loggers.plots import MlflowPlotLogger
from artifact_experiment._impl.backends.mlflow.loggers.score_collections import (
    MlflowScoreCollectionLogger,
)
from artifact_experiment._impl.backends.mlflow.loggers.scores import MlflowScoreLogger


class MlflowLoggingWorker(BackendLoggingWorker[MlflowRunAdapter]):
    @staticmethod
    def _get_score_logger(
        run: MlflowRunAdapter,
    ) -> ArtifactLogger[MlflowRunAdapter, Score]:
        return MlflowScoreLogger(run=run)

    @staticmethod
    def _get_array_logger(
        run: MlflowRunAdapter,
    ) -> ArtifactLogger[MlflowRunAdapter, Array]:
        return MlflowArrayLogger(run=run)

    @staticmethod
    def _get_plot_logger(
        run: MlflowRunAdapter,
    ) -> ArtifactLogger[MlflowRunAdapter, Plot]:
        return MlflowPlotLogger(run=run)

    @staticmethod
    def _get_score_collection_logger(
        run: MlflowRunAdapter,
    ) -> ArtifactLogger[MlflowRunAdapter, ScoreCollection]:
        return MlflowScoreCollectionLogger(run=run)

    @staticmethod
    def _get_array_collection_logger(
        run: MlflowRunAdapter,
    ) -> ArtifactLogger[MlflowRunAdapter, ArrayCollection]:
        return MlflowArrayCollectionLogger(run=run)

    @staticmethod
    def _get_plot_collection_logger(
        run: MlflowRunAdapter,
    ) -> ArtifactLogger[MlflowRunAdapter, PlotCollection]:
        return MlflowPlotCollectionLogger(run=run)

    @staticmethod
    def _get_file_logger(
        run: MlflowRunAdapter,
    ) -> BackendLogger[MlflowRunAdapter, File]:
        return MlflowFileLogger(run=run)
