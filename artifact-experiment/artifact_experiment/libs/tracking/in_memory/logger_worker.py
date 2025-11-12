from typing import Dict

from artifact_experiment.base.entities.file import File
from artifact_experiment.base.tracking.backend.logger import ArtifactLogger, BackendLogger
from artifact_experiment.base.tracking.backend.logger_worker import LoggerWorker
from artifact_experiment.libs.tracking.in_memory.adapter import InMemoryRunAdapter
from artifact_experiment.libs.tracking.in_memory.loggers.array_collections import (
    InMemoryArrayCollectionLogger,
)
from artifact_experiment.libs.tracking.in_memory.loggers.arrays import InMemoryArrayLogger
from artifact_experiment.libs.tracking.in_memory.loggers.files import InMemoryFileLogger
from artifact_experiment.libs.tracking.in_memory.loggers.plot_collections import (
    InMemoryPlotCollectionLogger,
)
from artifact_experiment.libs.tracking.in_memory.loggers.plots import InMemoryPlotLogger
from artifact_experiment.libs.tracking.in_memory.loggers.score_collections import (
    InMemoryScoreCollectionLogger,
)
from artifact_experiment.libs.tracking.in_memory.loggers.scores import InMemoryScoreLogger


class InMemoryLoggerWorker(LoggerWorker[InMemoryRunAdapter]):
    @staticmethod
    def _get_score_logger(
        run: InMemoryRunAdapter,
    ) -> ArtifactLogger[float, InMemoryRunAdapter]:
        return InMemoryScoreLogger(run=run)

    @staticmethod
    def _get_array_logger(
        run: InMemoryRunAdapter,
    ) -> ArtifactLogger[Array, InMemoryRunAdapter]:
        return InMemoryArrayLogger(run=run)

    @staticmethod
    def _get_plot_logger(
        run: InMemoryRunAdapter,
    ) -> ArtifactLogger[Figure, InMemoryRunAdapter]:
        return InMemoryPlotLogger(run=run)

    @staticmethod
    def _get_score_collection_logger(
        run: InMemoryRunAdapter,
    ) -> ArtifactLogger[Dict[str, float], InMemoryRunAdapter]:
        return InMemoryScoreCollectionLogger(run=run)

    @staticmethod
    def _get_array_collection_logger(
        run: InMemoryRunAdapter,
    ) -> ArtifactLogger[Dict[str, Array], InMemoryRunAdapter]:
        return InMemoryArrayCollectionLogger(run=run)

    @staticmethod
    def _get_plot_collection_logger(
        run: InMemoryRunAdapter,
    ) -> ArtifactLogger[Dict[str, Figure], InMemoryRunAdapter]:
        return InMemoryPlotCollectionLogger(run=run)

    @staticmethod
    def _get_file_logger(
        run: InMemoryRunAdapter,
    ) -> BackendLogger[File, InMemoryRunAdapter]:
        return InMemoryFileLogger(run=run)
