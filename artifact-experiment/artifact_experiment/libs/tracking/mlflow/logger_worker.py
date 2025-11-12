from typing import Dict

from artifact_experiment.base.entities.file import File
from artifact_experiment.base.tracking.backend.logger import ArtifactLogger, BackendLogger
from artifact_experiment.base.tracking.backend.logger_worker import LoggerWorker
from artifact_experiment.libs.tracking.mlflow.adapter import MlflowRunAdapter
from artifact_experiment.libs.tracking.mlflow.loggers.array_collections import (
    MlflowArrayCollectionLogger,
)
from artifact_experiment.libs.tracking.mlflow.loggers.arrays import MlflowArrayLogger
from artifact_experiment.libs.tracking.mlflow.loggers.files import MlflowFileLogger
from artifact_experiment.libs.tracking.mlflow.loggers.plot_collections import (
    MlflowPlotCollectionLogger,
)
from artifact_experiment.libs.tracking.mlflow.loggers.plots import MlflowPlotLogger
from artifact_experiment.libs.tracking.mlflow.loggers.score_collections import (
    MlflowScoreCollectionLogger,
)
from artifact_experiment.libs.tracking.mlflow.loggers.scores import MlflowScoreLogger


class MlflowLoggerWorker(LoggerWorker[MlflowRunAdapter]):
    @staticmethod
    def _get_score_logger(
        run: MlflowRunAdapter,
    ) -> ArtifactLogger[float, MlflowRunAdapter]:
        return MlflowScoreLogger(run=run)

    @staticmethod
    def _get_array_logger(
        run: MlflowRunAdapter,
    ) -> ArtifactLogger[Array, MlflowRunAdapter]:
        return MlflowArrayLogger(run=run)

    @staticmethod
    def _get_plot_logger(
        run: MlflowRunAdapter,
    ) -> ArtifactLogger[Figure, MlflowRunAdapter]:
        return MlflowPlotLogger(run=run)

    @staticmethod
    def _get_score_collection_logger(
        run: MlflowRunAdapter,
    ) -> ArtifactLogger[Dict[str, float], MlflowRunAdapter]:
        return MlflowScoreCollectionLogger(run=run)

    @staticmethod
    def _get_array_collection_logger(
        run: MlflowRunAdapter,
    ) -> ArtifactLogger[Dict[str, Array], MlflowRunAdapter]:
        return MlflowArrayCollectionLogger(run=run)

    @staticmethod
    def _get_plot_collection_logger(
        run: MlflowRunAdapter,
    ) -> ArtifactLogger[Dict[str, Figure], MlflowRunAdapter]:
        return MlflowPlotCollectionLogger(run=run)

    @staticmethod
    def _get_file_logger(
        run: MlflowRunAdapter,
    ) -> BackendLogger[File, MlflowRunAdapter]:
        return MlflowFileLogger(run=run)
