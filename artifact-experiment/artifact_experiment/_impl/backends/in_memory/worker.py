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
from artifact_experiment._impl.backends.in_memory.adapter import InMemoryRunAdapter
from artifact_experiment._impl.backends.in_memory.loggers.array_collections import (
    InMemoryArrayCollectionLogger,
)
from artifact_experiment._impl.backends.in_memory.loggers.arrays import InMemoryArrayLogger
from artifact_experiment._impl.backends.in_memory.loggers.files import InMemoryFileLogger
from artifact_experiment._impl.backends.in_memory.loggers.plot_collections import (
    InMemoryPlotCollectionLogger,
)
from artifact_experiment._impl.backends.in_memory.loggers.plots import InMemoryPlotLogger
from artifact_experiment._impl.backends.in_memory.loggers.score_collections import (
    InMemoryScoreCollectionLogger,
)
from artifact_experiment._impl.backends.in_memory.loggers.scores import InMemoryScoreLogger


class InMemoryLoggingWorker(BackendLoggingWorker[InMemoryRunAdapter]):
    @staticmethod
    def _get_score_logger(
        run: InMemoryRunAdapter,
    ) -> ArtifactLogger[InMemoryRunAdapter, Score]:
        return InMemoryScoreLogger(run=run)

    @staticmethod
    def _get_array_logger(
        run: InMemoryRunAdapter,
    ) -> ArtifactLogger[InMemoryRunAdapter, Array]:
        return InMemoryArrayLogger(run=run)

    @staticmethod
    def _get_plot_logger(
        run: InMemoryRunAdapter,
    ) -> ArtifactLogger[InMemoryRunAdapter, Plot]:
        return InMemoryPlotLogger(run=run)

    @staticmethod
    def _get_score_collection_logger(
        run: InMemoryRunAdapter,
    ) -> ArtifactLogger[InMemoryRunAdapter, ScoreCollection]:
        return InMemoryScoreCollectionLogger(run=run)

    @staticmethod
    def _get_array_collection_logger(
        run: InMemoryRunAdapter,
    ) -> ArtifactLogger[InMemoryRunAdapter, ArrayCollection]:
        return InMemoryArrayCollectionLogger(run=run)

    @staticmethod
    def _get_plot_collection_logger(
        run: InMemoryRunAdapter,
    ) -> ArtifactLogger[InMemoryRunAdapter, PlotCollection]:
        return InMemoryPlotCollectionLogger(run=run)

    @staticmethod
    def _get_file_logger(
        run: InMemoryRunAdapter,
    ) -> BackendLogger[InMemoryRunAdapter, File]:
        return InMemoryFileLogger(run=run)
