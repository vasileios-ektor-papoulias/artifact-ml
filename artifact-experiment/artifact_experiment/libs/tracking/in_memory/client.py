from typing import Dict, Optional, Type, TypeVar

import numpy as np
from matplotlib.figure import Figure

from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryNativeRun,
    InMemoryTrackingAdapter,
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


class InMemoryTrackingClient(TrackingClient[InMemoryTrackingAdapter]):
    @classmethod
    def build(
        cls: Type[InMemoryTrackingClientT], experiment_id: str, run_id: Optional[str] = None
    ) -> InMemoryTrackingClientT:
        run = InMemoryTrackingAdapter.build(experiment_id=experiment_id, run_id=run_id)
        client = cls._build(run=run)
        return client

    @classmethod
    def from_native_run(
        cls: Type[InMemoryTrackingClientT], native_run: InMemoryNativeRun
    ) -> InMemoryTrackingClientT:
        run = InMemoryTrackingAdapter.from_native_run(native_run=native_run)
        client = cls._build(run=run)
        return client

    @property
    def uploaded_files(self):
        return self._run.uploaded_files

    @staticmethod
    def _get_score_logger(
        run: InMemoryTrackingAdapter,
    ) -> ArtifactLogger[float, InMemoryTrackingAdapter]:
        return InMemoryScoreLogger(run=run)

    @staticmethod
    def _get_array_logger(
        run: InMemoryTrackingAdapter,
    ) -> ArtifactLogger[np.ndarray, InMemoryTrackingAdapter]:
        return InMemoryArrayLogger(run=run)

    @staticmethod
    def _get_plot_logger(
        run: InMemoryTrackingAdapter,
    ) -> ArtifactLogger[Figure, InMemoryTrackingAdapter]:
        return InMemoryPlotLogger(run=run)

    @staticmethod
    def _get_score_collection_logger(
        run: InMemoryTrackingAdapter,
    ) -> ArtifactLogger[Dict[str, float], InMemoryTrackingAdapter]:
        return InMemoryScoreCollectionLogger(run=run)

    @staticmethod
    def _get_array_collection_logger(
        run: InMemoryTrackingAdapter,
    ) -> ArtifactLogger[Dict[str, np.ndarray], InMemoryTrackingAdapter]:
        return InMemoryArrayCollectionLogger(run=run)

    @staticmethod
    def _get_plot_collection_logger(
        run: InMemoryTrackingAdapter,
    ) -> ArtifactLogger[Dict[str, Figure], InMemoryTrackingAdapter]:
        return InMemoryPlotCollectionLogger(run=run)
