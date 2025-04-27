from typing import Dict, Optional, Type

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.libs.tracking.filesystem.backend import (
    FilesystemBackend,
)
from artifact_experiment.libs.tracking.filesystem.loggers.array_collections import (
    FilesystemArrayCollectionLogger,
)
from artifact_experiment.libs.tracking.filesystem.loggers.arrays import FilesystemArrayLogger
from artifact_experiment.libs.tracking.filesystem.loggers.plot_collections import (
    FilesystemPlotCollectionLogger,
)
from artifact_experiment.libs.tracking.filesystem.loggers.plots import FilesystemPlotLogger
from artifact_experiment.libs.tracking.filesystem.loggers.score_collections import (
    FilesystemScoreCollectionLogger,
)
from artifact_experiment.libs.tracking.filesystem.loggers.scores import FilesystemScoreLogger


class FilesystemTrackingClient(TrackingClient[FilesystemBackend]):
    @classmethod
    def build(
        cls: Type["FilesystemTrackingClient"], experiment_id: Optional[str] = None
    ) -> "FilesystemTrackingClient":
        backend = FilesystemBackend.build(experiment_id=experiment_id)
        client = FilesystemTrackingClient(backend=backend)
        return client

    @staticmethod
    def _get_score_logger(backend: FilesystemBackend) -> ArtifactLogger[float, FilesystemBackend]:
        return FilesystemScoreLogger(backend=backend)

    @staticmethod
    def _get_array_logger(
        backend: FilesystemBackend,
    ) -> ArtifactLogger[ndarray, FilesystemBackend]:
        return FilesystemArrayLogger(backend=backend)

    @staticmethod
    def _get_plot_logger(backend: FilesystemBackend) -> ArtifactLogger[Figure, FilesystemBackend]:
        return FilesystemPlotLogger(backend=backend)

    @staticmethod
    def _get_score_collection_logger(
        backend: FilesystemBackend,
    ) -> ArtifactLogger[Dict[str, float], FilesystemBackend]:
        return FilesystemScoreCollectionLogger(backend=backend)

    @staticmethod
    def _get_array_collection_logger(
        backend: FilesystemBackend,
    ) -> ArtifactLogger[Dict[str, ndarray], FilesystemBackend]:
        return FilesystemArrayCollectionLogger(backend=backend)

    @staticmethod
    def _get_plot_collection_logger(
        backend: FilesystemBackend,
    ) -> ArtifactLogger[Dict[str, Figure], FilesystemBackend]:
        return FilesystemPlotCollectionLogger(backend=backend)
