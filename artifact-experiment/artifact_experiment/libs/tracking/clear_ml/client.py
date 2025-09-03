from typing import Dict, Optional, Type, TypeVar

from clearml import Task
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.libs.tracking.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment.libs.tracking.clear_ml.loggers.array_collections import (
    ClearMLArrayCollectionLogger,
)
from artifact_experiment.libs.tracking.clear_ml.loggers.arrays import ClearMLArrayLogger
from artifact_experiment.libs.tracking.clear_ml.loggers.plot_collections import (
    ClearMLPlotCollectionLogger,
)
from artifact_experiment.libs.tracking.clear_ml.loggers.plots import ClearMLPlotLogger
from artifact_experiment.libs.tracking.clear_ml.loggers.score_collections import (
    ClearMLScoreCollectionLogger,
)
from artifact_experiment.libs.tracking.clear_ml.loggers.scores import ClearMLScoreLogger

ClearMLTrackingClientT = TypeVar("ClearMLTrackingClientT", bound="ClearMLTrackingClient")


class ClearMLTrackingClient(TrackingClient[ClearMLRunAdapter]):
    @classmethod
    def build(
        cls: Type[ClearMLTrackingClientT], experiment_id: str, run_id: Optional[str] = None
    ) -> ClearMLTrackingClientT:
        run = ClearMLRunAdapter.build(experiment_id=experiment_id, run_id=run_id)
        client = cls._build(run=run)
        return client

    @classmethod
    def from_native_run(
        cls: Type[ClearMLTrackingClientT], native_run: Task
    ) -> ClearMLTrackingClientT:
        run = ClearMLRunAdapter.from_native_run(native_run=native_run)
        client = cls._build(run=run)
        return client

    @staticmethod
    def _get_score_logger(run: ClearMLRunAdapter) -> ArtifactLogger[float, ClearMLRunAdapter]:
        return ClearMLScoreLogger(run=run)

    @staticmethod
    def _get_array_logger(
        run: ClearMLRunAdapter,
    ) -> ArtifactLogger[ndarray, ClearMLRunAdapter]:
        return ClearMLArrayLogger(run=run)

    @staticmethod
    def _get_plot_logger(run: ClearMLRunAdapter) -> ArtifactLogger[Figure, ClearMLRunAdapter]:
        return ClearMLPlotLogger(run=run)

    @staticmethod
    def _get_score_collection_logger(
        run: ClearMLRunAdapter,
    ) -> ArtifactLogger[Dict[str, float], ClearMLRunAdapter]:
        return ClearMLScoreCollectionLogger(run=run)

    @staticmethod
    def _get_array_collection_logger(
        run: ClearMLRunAdapter,
    ) -> ArtifactLogger[Dict[str, ndarray], ClearMLRunAdapter]:
        return ClearMLArrayCollectionLogger(run=run)

    @staticmethod
    def _get_plot_collection_logger(
        run: ClearMLRunAdapter,
    ) -> ArtifactLogger[Dict[str, Figure], ClearMLRunAdapter]:
        return ClearMLPlotCollectionLogger(run=run)
