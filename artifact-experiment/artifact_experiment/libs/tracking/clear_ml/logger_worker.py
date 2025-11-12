from typing import Dict

from artifact_experiment.base.entities.file import File
from artifact_experiment.base.tracking.backend.logger import ArtifactLogger, BackendLogger
from artifact_experiment.base.tracking.backend.logger_worker import LoggerWorker
from artifact_experiment.libs.tracking.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment.libs.tracking.clear_ml.loggers.array_collections import (
    ClearMLArrayCollectionLogger,
)
from artifact_experiment.libs.tracking.clear_ml.loggers.arrays import ClearMLArrayLogger
from artifact_experiment.libs.tracking.clear_ml.loggers.files import ClearMLFileLogger
from artifact_experiment.libs.tracking.clear_ml.loggers.plot_collections import (
    ClearMLPlotCollectionLogger,
)
from artifact_experiment.libs.tracking.clear_ml.loggers.plots import ClearMLPlotLogger
from artifact_experiment.libs.tracking.clear_ml.loggers.score_collections import (
    ClearMLScoreCollectionLogger,
)
from artifact_experiment.libs.tracking.clear_ml.loggers.scores import ClearMLScoreLogger


class ClearMLLoggerWorker(LoggerWorker[ClearMLRunAdapter]):
    @staticmethod
    def _get_score_logger(
        run: ClearMLRunAdapter,
    ) -> ArtifactLogger[float, ClearMLRunAdapter]:
        return ClearMLScoreLogger(run=run)

    @staticmethod
    def _get_array_logger(
        run: ClearMLRunAdapter,
    ) -> ArtifactLogger[Array, ClearMLRunAdapter]:
        return ClearMLArrayLogger(run=run)

    @staticmethod
    def _get_plot_logger(
        run: ClearMLRunAdapter,
    ) -> ArtifactLogger[Figure, ClearMLRunAdapter]:
        return ClearMLPlotLogger(run=run)

    @staticmethod
    def _get_score_collection_logger(
        run: ClearMLRunAdapter,
    ) -> ArtifactLogger[Dict[str, float], ClearMLRunAdapter]:
        return ClearMLScoreCollectionLogger(run=run)

    @staticmethod
    def _get_array_collection_logger(
        run: ClearMLRunAdapter,
    ) -> ArtifactLogger[Dict[str, Array], ClearMLRunAdapter]:
        return ClearMLArrayCollectionLogger(run=run)

    @staticmethod
    def _get_plot_collection_logger(
        run: ClearMLRunAdapter,
    ) -> ArtifactLogger[Dict[str, Figure], ClearMLRunAdapter]:
        return ClearMLPlotCollectionLogger(run=run)

    @staticmethod
    def _get_file_logger(
        run: ClearMLRunAdapter,
    ) -> BackendLogger[File, ClearMLRunAdapter]:
        return ClearMLFileLogger(run=run)
