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
from artifact_experiment._impl.backends.neptune.adapter import NeptuneRunAdapter
from artifact_experiment._impl.backends.neptune.loggers.array_collections import (
    NeptuneArrayCollectionLogger,
)
from artifact_experiment._impl.backends.neptune.loggers.arrays import NeptuneArrayLogger
from artifact_experiment._impl.backends.neptune.loggers.files import NeptuneFileLogger
from artifact_experiment._impl.backends.neptune.loggers.plot_collections import (
    NeptunePlotCollectionLogger,
)
from artifact_experiment._impl.backends.neptune.loggers.plots import NeptunePlotLogger
from artifact_experiment._impl.backends.neptune.loggers.score_collections import (
    NeptuneScoreCollectionLogger,
)
from artifact_experiment._impl.backends.neptune.loggers.scores import NeptuneScoreLogger


class NeptuneLoggingWorker(BackendLoggingWorker[NeptuneRunAdapter]):
    @staticmethod
    def _get_score_logger(
        run: NeptuneRunAdapter,
    ) -> ArtifactLogger[NeptuneRunAdapter, Score]:
        return NeptuneScoreLogger(run=run)

    @staticmethod
    def _get_array_logger(
        run: NeptuneRunAdapter,
    ) -> ArtifactLogger[NeptuneRunAdapter, Array]:
        return NeptuneArrayLogger(run=run)

    @staticmethod
    def _get_plot_logger(
        run: NeptuneRunAdapter,
    ) -> ArtifactLogger[NeptuneRunAdapter, Plot]:
        return NeptunePlotLogger(run=run)

    @staticmethod
    def _get_score_collection_logger(
        run: NeptuneRunAdapter,
    ) -> ArtifactLogger[NeptuneRunAdapter, ScoreCollection]:
        return NeptuneScoreCollectionLogger(run=run)

    @staticmethod
    def _get_array_collection_logger(
        run: NeptuneRunAdapter,
    ) -> ArtifactLogger[NeptuneRunAdapter, ArrayCollection]:
        return NeptuneArrayCollectionLogger(run=run)

    @staticmethod
    def _get_plot_collection_logger(
        run: NeptuneRunAdapter,
    ) -> ArtifactLogger[NeptuneRunAdapter, PlotCollection]:
        return NeptunePlotCollectionLogger(run=run)

    @staticmethod
    def _get_file_logger(
        run: NeptuneRunAdapter,
    ) -> BackendLogger[NeptuneRunAdapter, File]:
        return NeptuneFileLogger(run=run)
