from typing import Dict

import numpy as np
from matplotlib.figure import Figure

from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.libs.tracking.in_memory.adapter import (
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


class InMemoryTrackingClient(TrackingClient[InMemoryTrackingAdapter]):
    def __init__(self, run: InMemoryTrackingAdapter):
        super().__init__(run)

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
