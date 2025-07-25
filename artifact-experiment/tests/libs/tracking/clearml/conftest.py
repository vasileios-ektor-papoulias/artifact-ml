from typing import Callable, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from artifact_experiment.libs.tracking.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment.libs.tracking.clear_ml.client import ClearMLTrackingClient
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
from artifact_experiment.libs.tracking.clear_ml.stores.files import ClearMLFileStore
from artifact_experiment.libs.tracking.clear_ml.stores.plots import ClearMLPlotStore
from artifact_experiment.libs.tracking.clear_ml.stores.scores import ClearMLScoreStore
from clearml import Task


@pytest.fixture
def native_run_factory(mocker) -> Callable[[Optional[str], Optional[str]], MagicMock]:
    def _factory(
        experiment_id: Optional[str] = "default_project",
        run_id: Optional[str] = "default_task",
    ) -> MagicMock:
        task = mocker.MagicMock(spec=Task)
        task.id = f"{experiment_id}_{run_id}"
        task.name = run_id
        task.project = experiment_id
        task.get_status.return_value = "in_progress"
        task.get_logger.return_value = mocker.MagicMock()
        task.artifacts = {}
        task.get_reported_scalars.return_value = {}
        task.get_reported_plots.return_value = {}

        return task

    return _factory


@pytest.fixture
def adapter_factory(
    native_run_factory: Callable[[Optional[str], Optional[str]], MagicMock],
) -> Callable[[Optional[str], Optional[str]], ClearMLRunAdapter]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> ClearMLRunAdapter:
        native_run = native_run_factory(experiment_id, run_id)
        adapter = ClearMLRunAdapter(native_run=native_run)
        return adapter

    return _factory


@pytest.fixture
def loggers_factory(
    adapter_factory: Callable[[Optional[str], Optional[str]], ClearMLRunAdapter],
) -> Callable[
    [Optional[str], Optional[str]],
    Tuple[ClearMLRunAdapter, ClearMLScoreStore, ClearMLPlotStore, ClearMLFileStore],
]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[ClearMLRunAdapter, ClearMLScoreStore, ClearMLPlotStore, ClearMLFileStore]:
        adapter = adapter_factory(experiment_id, run_id)

        score_store = adapter.get_exported_scores()
        plot_store = adapter.get_exported_plots()
        file_store = adapter.get_exported_files()

        return adapter, score_store, plot_store, file_store

    return _factory


@pytest.fixture
def loggers_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MagicMock, ClearMLRunAdapter],
    ],
) -> Callable[
    [Optional[str], Optional[str]],
    Tuple[
        ClearMLRunAdapter,
        ClearMLScoreLogger,
        ClearMLArrayLogger,
        ClearMLPlotLogger,
        ClearMLScoreCollectionLogger,
        ClearMLArrayCollectionLogger,
        ClearMLPlotCollectionLogger,
    ],
]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[
        ClearMLRunAdapter,
        ClearMLScoreLogger,
        ClearMLArrayLogger,
        ClearMLPlotLogger,
        ClearMLScoreCollectionLogger,
        ClearMLArrayCollectionLogger,
        ClearMLPlotCollectionLogger,
    ]:
        _, adapter = adapter_factory(experiment_id, run_id)
        score_logger = ClearMLScoreLogger(run=adapter)
        array_logger = ClearMLArrayLogger(run=adapter)
        plot_logger = ClearMLPlotLogger(run=adapter)
        score_collection_logger = ClearMLScoreCollectionLogger(run=adapter)
        array_collection_logger = ClearMLArrayCollectionLogger(run=adapter)
        plot_collection_logger = ClearMLPlotCollectionLogger(run=adapter)
        return (
            adapter,
            score_logger,
            array_logger,
            plot_logger,
            score_collection_logger,
            array_collection_logger,
            plot_collection_logger,
        )

    return _factory


@pytest.fixture
def client_factory(
    loggers_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[
            ClearMLRunAdapter,
            ClearMLScoreLogger,
            ClearMLArrayLogger,
            ClearMLPlotLogger,
            ClearMLScoreCollectionLogger,
            ClearMLArrayCollectionLogger,
            ClearMLPlotCollectionLogger,
        ],
    ],
) -> Callable[
    [Optional[str], Optional[str]],
    Tuple[
        ClearMLRunAdapter,
        ClearMLScoreLogger,
        ClearMLArrayLogger,
        ClearMLPlotLogger,
        ClearMLScoreCollectionLogger,
        ClearMLArrayCollectionLogger,
        ClearMLPlotCollectionLogger,
        ClearMLTrackingClient,
    ],
]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[
        ClearMLRunAdapter,
        ClearMLScoreLogger,
        ClearMLArrayLogger,
        ClearMLPlotLogger,
        ClearMLScoreCollectionLogger,
        ClearMLArrayCollectionLogger,
        ClearMLPlotCollectionLogger,
        ClearMLTrackingClient,
    ]:
        (
            adapter,
            score_logger,
            array_logger,
            plot_logger,
            score_collection_logger,
            array_collection_logger,
            plot_collection_logger,
        ) = loggers_factory(experiment_id, run_id)
        client = ClearMLTrackingClient(
            run=adapter,
            score_logger=score_logger,
            array_logger=array_logger,
            plot_logger=plot_logger,
            score_collection_logger=score_collection_logger,
            array_collection_logger=array_collection_logger,
            plot_collection_logger=plot_collection_logger,
        )
        return (
            adapter,
            score_logger,
            array_logger,
            plot_logger,
            score_collection_logger,
            array_collection_logger,
            plot_collection_logger,
            client,
        )

    return _factory
