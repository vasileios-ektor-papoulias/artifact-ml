from typing import Dict

from artifact_experiment.base.entities.file import File
from artifact_experiment.base.tracking.backend.logger import ArtifactLogger, BackendLogger
from artifact_experiment.base.tracking.backend.logger_worker import LoggerWorker
from artifact_experiment.libs.tracking.filesystem.adapter import FilesystemRunAdapter
from artifact_experiment.libs.tracking.filesystem.loggers.array_collections import (
    FilesystemArrayCollectionLogger,
)
from artifact_experiment.libs.tracking.filesystem.loggers.arrays import FilesystemArrayLogger
from artifact_experiment.libs.tracking.filesystem.loggers.files import FilesystemFileLogger
from artifact_experiment.libs.tracking.filesystem.loggers.plot_collections import (
    FilesystemPlotCollectionLogger,
)
from artifact_experiment.libs.tracking.filesystem.loggers.plots import FilesystemPlotLogger
from artifact_experiment.libs.tracking.filesystem.loggers.score_collections import (
    FilesystemScoreCollectionLogger,
)
from artifact_experiment.libs.tracking.filesystem.loggers.scores import FilesystemScoreLogger


class FilesystemLoggerWorker(LoggerWorker[FilesystemRunAdapter]):
    @staticmethod
    def _get_score_logger(
        run: FilesystemRunAdapter,
    ) -> ArtifactLogger[float, FilesystemRunAdapter]:
        return FilesystemScoreLogger(run=run)

    @staticmethod
    def _get_array_logger(
        run: FilesystemRunAdapter,
    ) -> ArtifactLogger[Array, FilesystemRunAdapter]:
        return FilesystemArrayLogger(run=run)

    @staticmethod
    def _get_plot_logger(
        run: FilesystemRunAdapter,
    ) -> ArtifactLogger[Figure, FilesystemRunAdapter]:
        return FilesystemPlotLogger(run=run)

    @staticmethod
    def _get_score_collection_logger(
        run: FilesystemRunAdapter,
    ) -> ArtifactLogger[Dict[str, float], FilesystemRunAdapter]:
        return FilesystemScoreCollectionLogger(run=run)

    @staticmethod
    def _get_array_collection_logger(
        run: FilesystemRunAdapter,
    ) -> ArtifactLogger[Dict[str, Array], FilesystemRunAdapter]:
        return FilesystemArrayCollectionLogger(run=run)

    @staticmethod
    def _get_plot_collection_logger(
        run: FilesystemRunAdapter,
    ) -> ArtifactLogger[Dict[str, Figure], FilesystemRunAdapter]:
        return FilesystemPlotCollectionLogger(run=run)

    @staticmethod
    def _get_file_logger(
        run: FilesystemRunAdapter,
    ) -> BackendLogger[File, FilesystemRunAdapter]:
        return FilesystemFileLogger(run=run)
