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
from artifact_experiment._impl.backends.filesystem.adapter import FilesystemRunAdapter
from artifact_experiment._impl.backends.filesystem.loggers.array_collections import (
    FilesystemArrayCollectionLogger,
)
from artifact_experiment._impl.backends.filesystem.loggers.arrays import FilesystemArrayLogger
from artifact_experiment._impl.backends.filesystem.loggers.files import FilesystemFileLogger
from artifact_experiment._impl.backends.filesystem.loggers.plot_collections import (
    FilesystemPlotCollectionLogger,
)
from artifact_experiment._impl.backends.filesystem.loggers.plots import FilesystemPlotLogger
from artifact_experiment._impl.backends.filesystem.loggers.score_collections import (
    FilesystemScoreCollectionLogger,
)
from artifact_experiment._impl.backends.filesystem.loggers.scores import FilesystemScoreLogger


class FilesystemLoggingWorker(BackendLoggingWorker[FilesystemRunAdapter]):
    @staticmethod
    def _get_score_logger(
        run: FilesystemRunAdapter,
    ) -> ArtifactLogger[FilesystemRunAdapter, Score]:
        return FilesystemScoreLogger(run=run)

    @staticmethod
    def _get_array_logger(
        run: FilesystemRunAdapter,
    ) -> ArtifactLogger[FilesystemRunAdapter, Array]:
        return FilesystemArrayLogger(run=run)

    @staticmethod
    def _get_plot_logger(
        run: FilesystemRunAdapter,
    ) -> ArtifactLogger[FilesystemRunAdapter, Plot]:
        return FilesystemPlotLogger(run=run)

    @staticmethod
    def _get_score_collection_logger(
        run: FilesystemRunAdapter,
    ) -> ArtifactLogger[FilesystemRunAdapter, ScoreCollection]:
        return FilesystemScoreCollectionLogger(run=run)

    @staticmethod
    def _get_array_collection_logger(
        run: FilesystemRunAdapter,
    ) -> ArtifactLogger[FilesystemRunAdapter, ArrayCollection]:
        return FilesystemArrayCollectionLogger(run=run)

    @staticmethod
    def _get_plot_collection_logger(
        run: FilesystemRunAdapter,
    ) -> ArtifactLogger[FilesystemRunAdapter, PlotCollection]:
        return FilesystemPlotCollectionLogger(run=run)

    @staticmethod
    def _get_file_logger(
        run: FilesystemRunAdapter,
    ) -> BackendLogger[FilesystemRunAdapter, File]:
        return FilesystemFileLogger(run=run)
