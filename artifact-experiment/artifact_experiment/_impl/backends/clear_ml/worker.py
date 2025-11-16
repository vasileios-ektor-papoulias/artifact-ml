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
from artifact_experiment._impl.backends.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment._impl.backends.clear_ml.loggers.array_collections import (
    ClearMLArrayCollectionLogger,
)
from artifact_experiment._impl.backends.clear_ml.loggers.arrays import ClearMLArrayLogger
from artifact_experiment._impl.backends.clear_ml.loggers.files import ClearMLFileLogger
from artifact_experiment._impl.backends.clear_ml.loggers.plot_collections import (
    ClearMLPlotCollectionLogger,
)
from artifact_experiment._impl.backends.clear_ml.loggers.plots import ClearMLPlotLogger
from artifact_experiment._impl.backends.clear_ml.loggers.score_collections import (
    ClearMLScoreCollectionLogger,
)
from artifact_experiment._impl.backends.clear_ml.loggers.scores import ClearMLScoreLogger


class ClearMLLoggingWorker(BackendLoggingWorker[ClearMLRunAdapter]):
    @staticmethod
    def _get_score_logger(run: ClearMLRunAdapter) -> ArtifactLogger[ClearMLRunAdapter, Score]:
        return ClearMLScoreLogger(run=run)

    @staticmethod
    def _get_array_logger(run: ClearMLRunAdapter) -> ArtifactLogger[ClearMLRunAdapter, Array]:
        return ClearMLArrayLogger(run=run)

    @staticmethod
    def _get_plot_logger(run: ClearMLRunAdapter) -> ArtifactLogger[ClearMLRunAdapter, Plot]:
        return ClearMLPlotLogger(run=run)

    @staticmethod
    def _get_score_collection_logger(
        run: ClearMLRunAdapter,
    ) -> ArtifactLogger[ClearMLRunAdapter, ScoreCollection]:
        return ClearMLScoreCollectionLogger(run=run)

    @staticmethod
    def _get_array_collection_logger(
        run: ClearMLRunAdapter,
    ) -> ArtifactLogger[ClearMLRunAdapter, ArrayCollection]:
        return ClearMLArrayCollectionLogger(run=run)

    @staticmethod
    def _get_plot_collection_logger(
        run: ClearMLRunAdapter,
    ) -> ArtifactLogger[ClearMLRunAdapter, PlotCollection]:
        return ClearMLPlotCollectionLogger(run=run)

    @staticmethod
    def _get_file_logger(run: ClearMLRunAdapter) -> BackendLogger[ClearMLRunAdapter, File]:
        return ClearMLFileLogger(run=run)
