from typing import Dict, Optional, Type, TypeVar

import numpy as np
from matplotlib.figure import Figure

from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryRun,
    InMemoryRunAdapter,
)
from artifact_experiment.libs.tracking.in_memory.loggers.array_collections import (
    InMemoryArrayCollectionLogger,
)
from artifact_experiment.libs.tracking.in_memory.loggers.arrays import (
    InMemoryArrayLogger,
)
from artifact_experiment.libs.tracking.in_memory.loggers.plot_collections import (
    InMemoryPlotCollectionLogger,
)
from artifact_experiment.libs.tracking.in_memory.loggers.plots import (
    InMemoryPlotLogger,
)
from artifact_experiment.libs.tracking.in_memory.loggers.score_collections import (
    InMemoryScoreCollectionLogger,
)
from artifact_experiment.libs.tracking.in_memory.loggers.scores import (
    InMemoryScoreLogger,
)

InMemoryTrackingClientT = TypeVar("InMemoryTrackingClientT", bound="InMemoryTrackingClient")


class InMemoryTrackingClient(TrackingClient[InMemoryRunAdapter]):
    @classmethod
    def build(
        cls: Type[InMemoryTrackingClientT], experiment_id: str, run_id: Optional[str] = None
    ) -> InMemoryTrackingClientT:
        run = InMemoryRunAdapter.build(experiment_id=experiment_id, run_id=run_id)
        client = cls._build(run=run)
        return client

    @classmethod
    def from_native_run(
        cls: Type[InMemoryTrackingClientT], native_run: InMemoryRun
    ) -> InMemoryTrackingClientT:
        run = InMemoryRunAdapter.from_native_run(native_run=native_run)
        client = cls._build(run=run)
        return client

    @property
    def uploaded_files(self):
        return self._run.uploaded_files

    @staticmethod
    def _get_score_logger(
        run: InMemoryRunAdapter,
    ) -> ArtifactLogger[float, InMemoryRunAdapter]:
        return InMemoryScoreLogger(run=run)

    @staticmethod
    def _get_array_logger(
        run: InMemoryRunAdapter,
    ) -> ArtifactLogger[np.ndarray, InMemoryRunAdapter]:
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
    ) -> ArtifactLogger[Dict[str, np.ndarray], InMemoryRunAdapter]:
        return InMemoryArrayCollectionLogger(run=run)

    @staticmethod
    def _get_plot_collection_logger(
        run: InMemoryRunAdapter,
    ) -> ArtifactLogger[Dict[str, Figure], InMemoryRunAdapter]:
        return InMemoryPlotCollectionLogger(run=run)
