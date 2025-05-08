from typing import Dict, Optional, Type, TypeVar

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.libs.tracking.filesystem.adapter import (
    FilesystemRunAdapter,
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

flesystemTrackingClientT = TypeVar("flesystemTrackingClientT", bound="FilesystemTrackingClient")


class FilesystemTrackingClient(TrackingClient[FilesystemRunAdapter]):
    @classmethod
    def build(
        cls: Type[flesystemTrackingClientT], experiment_id: str, run_id: Optional[str] = None
    ) -> flesystemTrackingClientT:
        run = FilesystemRunAdapter.build(experiment_id=experiment_id, run_id=run_id)
        client = cls(run=run)
        return client

    @property
    def experiment_dir(self) -> str:
        return self._run.experiment_dir

    @property
    def run_dir(self) -> str:
        return self._run.dir

    @staticmethod
    def _get_score_logger(
        run: FilesystemRunAdapter,
    ) -> ArtifactLogger[float, FilesystemRunAdapter]:
        return FilesystemScoreLogger(run=run)

    @staticmethod
    def _get_array_logger(
        run: FilesystemRunAdapter,
    ) -> ArtifactLogger[ndarray, FilesystemRunAdapter]:
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
    ) -> ArtifactLogger[Dict[str, ndarray], FilesystemRunAdapter]:
        return FilesystemArrayCollectionLogger(run=run)

    @staticmethod
    def _get_plot_collection_logger(
        run: FilesystemRunAdapter,
    ) -> ArtifactLogger[Dict[str, Figure], FilesystemRunAdapter]:
        return FilesystemPlotCollectionLogger(run=run)
