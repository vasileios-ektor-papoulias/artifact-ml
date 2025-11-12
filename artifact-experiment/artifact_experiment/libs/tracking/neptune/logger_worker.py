from typing import Dict

from artifact_experiment.base.entities.file import File
from artifact_experiment.base.tracking.backend.logger import ArtifactLogger, BackendLogger
from artifact_experiment.base.tracking.backend.logger_worker import LoggerWorker
from artifact_experiment.libs.tracking.neptune.adapter import NeptuneRunAdapter
from artifact_experiment.libs.tracking.neptune.loggers.array_collections import (
    NeptuneArrayCollectionLogger,
)
from artifact_experiment.libs.tracking.neptune.loggers.arrays import NeptuneArrayLogger
from artifact_experiment.libs.tracking.neptune.loggers.files import NeptuneFileLogger
from artifact_experiment.libs.tracking.neptune.loggers.plot_collections import (
    NeptunePlotCollectionLogger,
)
from artifact_experiment.libs.tracking.neptune.loggers.plots import NeptunePlotLogger
from artifact_experiment.libs.tracking.neptune.loggers.score_collections import (
    NeptuneScoreCollectionLogger,
)
from artifact_experiment.libs.tracking.neptune.loggers.scores import NeptuneScoreLogger


class NeptuneLoggerWorker(LoggerWorker[NeptuneRunAdapter]):
    @staticmethod
    def _get_score_logger(
        run: NeptuneRunAdapter,
    ) -> ArtifactLogger[float, NeptuneRunAdapter]:
        return NeptuneScoreLogger(run=run)

    @staticmethod
    def _get_array_logger(
        run: NeptuneRunAdapter,
    ) -> ArtifactLogger[Array, NeptuneRunAdapter]:
        return NeptuneArrayLogger(run=run)

    @staticmethod
    def _get_plot_logger(
        run: NeptuneRunAdapter,
    ) -> ArtifactLogger[Figure, NeptuneRunAdapter]:
        return NeptunePlotLogger(run=run)

    @staticmethod
    def _get_score_collection_logger(
        run: NeptuneRunAdapter,
    ) -> ArtifactLogger[Dict[str, float], NeptuneRunAdapter]:
        return NeptuneScoreCollectionLogger(run=run)

    @staticmethod
    def _get_array_collection_logger(
        run: NeptuneRunAdapter,
    ) -> ArtifactLogger[Dict[str, Array], NeptuneRunAdapter]:
        return NeptuneArrayCollectionLogger(run=run)

    @staticmethod
    def _get_plot_collection_logger(
        run: NeptuneRunAdapter,
    ) -> ArtifactLogger[Dict[str, Figure], NeptuneRunAdapter]:
        return NeptunePlotCollectionLogger(run=run)

    @staticmethod
    def _get_file_logger(
        run: NeptuneRunAdapter,
    ) -> BackendLogger[File, NeptuneRunAdapter]:
        return NeptuneFileLogger(run=run)
