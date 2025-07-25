from typing import Callable, Dict, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from artifact_experiment.libs.tracking.mlflow.adapter import (
    MlflowNativeRun,
    MlflowRunAdapter,
)
from artifact_experiment.libs.tracking.mlflow.client import MlflowTrackingClient
from artifact_experiment.libs.tracking.mlflow.loggers.array_collections import (
    MlflowArrayCollectionLogger,
)
from artifact_experiment.libs.tracking.mlflow.loggers.arrays import MlflowArrayLogger
from artifact_experiment.libs.tracking.mlflow.loggers.plot_collections import (
    MlflowPlotCollectionLogger,
)
from artifact_experiment.libs.tracking.mlflow.loggers.plots import MlflowPlotLogger
from artifact_experiment.libs.tracking.mlflow.loggers.score_collections import (
    MlflowScoreCollectionLogger,
)
from artifact_experiment.libs.tracking.mlflow.loggers.scores import MlflowScoreLogger
from mlflow.entities import Run, RunStatus
from mlflow.tracking import MlflowClient
from pytest_mock import MockerFixture


@pytest.fixture
def native_run_factory(
    mocker,
) -> Callable[[Optional[str], Optional[str]], Dict[str, MagicMock]]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, MagicMock]:
        if experiment_id is None:
            experiment_id = "default_experiment_id"
        if run_id is None:
            run_id = "default_run_id"
        # Mock Run Info
        run_info = mocker.MagicMock()
        run_info.run_id = run_id
        run_info.experiment_id = experiment_id
        run_info.run_name = run_id
        run_info.status = RunStatus.to_string(RunStatus.RUNNING)

        # Mock Run
        run = mocker.MagicMock(spec=Run)
        run.info = run_info
        run.data = mocker.MagicMock()

        # Mock Client
        client = mocker.MagicMock(spec=MlflowClient)
        client.log_artifact = mocker.MagicMock()
        client.list_artifacts = mocker.MagicMock()
        client.log_metric = mocker.MagicMock()
        client.get_metric_history = mocker.MagicMock()
        client.set_terminated = mocker.MagicMock()
        client.get_experiment = mocker.MagicMock()
        client.get_run = mocker.MagicMock(return_value=run)
        client.search_runs = mocker.MagicMock(return_value=[run])
        return {
            "client": client,
            "run": run,
            "run_info": run_info,
        }

    return _factory


@pytest.fixture
def patch_mlflow_run_creation(
    mocker: MockerFixture,
    native_run_factory: Callable[[Optional[str], Optional[str]], MagicMock],
):
    mocked_client = mocker.MagicMock(name="MlflowClient")
    mocker.patch(
        "artifact_experiment.libs.tracking.mlflow.adapter.MlflowClient",
        return_value=mocked_client,
    )
    mocked_client.search_runs.return_value = []

    def create_run_side_effect(experiment_id: str, run_name: str):
        run = native_run_factory(experiment_id, run_name)
        run.info.experiment_id = experiment_id
        run.info.run_id = f"{experiment_id}_{run_name}"
        run.info.run_name = run_name
        run.info.status = "RUNNING"
        return run

    mocked_client.create_run.side_effect = create_run_side_effect

    def get_run_side_effect(run_id: str):
        run = MagicMock(name="run")
        run.info.run_id = run_id
        run.info.run_name = run_id.split("_")[-1]
        run.info.experiment_id = run_id.split("_")[0]
        run.info.status = "RUNNING"
        return run

    mocked_client.get_run.side_effect = get_run_side_effect


@pytest.fixture
def adapter_factory(
    native_run_factory: Callable[[Optional[str], Optional[str]], Dict[str, MagicMock]],
) -> Callable[[Optional[str], Optional[str]], Tuple[MlflowNativeRun, MlflowRunAdapter]]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[MlflowNativeRun, MlflowRunAdapter]:
        native_entities = native_run_factory(experiment_id, run_id)

        native_run = MlflowNativeRun(
            client=native_entities["client"],
            run=native_entities["run"],
        )
        adapter = MlflowRunAdapter(native_run=native_run)

        return native_run, adapter

    return _factory


@pytest.fixture
def loggers_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MagicMock, MlflowRunAdapter],
    ],
) -> Callable[
    [Optional[str], Optional[str]],
    Tuple[
        MlflowRunAdapter,
        MlflowScoreLogger,
        MlflowArrayLogger,
        MlflowPlotLogger,
        MlflowScoreCollectionLogger,
        MlflowArrayCollectionLogger,
        MlflowPlotCollectionLogger,
    ],
]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[
        MlflowRunAdapter,
        MlflowScoreLogger,
        MlflowArrayLogger,
        MlflowPlotLogger,
        MlflowScoreCollectionLogger,
        MlflowArrayCollectionLogger,
        MlflowPlotCollectionLogger,
    ]:
        _, adapter = adapter_factory(experiment_id, run_id)
        score_logger = MlflowScoreLogger(run=adapter)
        array_logger = MlflowArrayLogger(run=adapter)
        plot_logger = MlflowPlotLogger(run=adapter)
        score_collection_logger = MlflowScoreCollectionLogger(run=adapter)
        array_collection_logger = MlflowArrayCollectionLogger(run=adapter)
        plot_collection_logger = MlflowPlotCollectionLogger(run=adapter)
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
            MlflowRunAdapter,
            MlflowScoreLogger,
            MlflowArrayLogger,
            MlflowPlotLogger,
            MlflowScoreCollectionLogger,
            MlflowArrayCollectionLogger,
            MlflowPlotCollectionLogger,
        ],
    ],
) -> Callable[
    [Optional[str], Optional[str]],
    Tuple[
        MlflowRunAdapter,
        MlflowScoreLogger,
        MlflowArrayLogger,
        MlflowPlotLogger,
        MlflowScoreCollectionLogger,
        MlflowArrayCollectionLogger,
        MlflowPlotCollectionLogger,
        MlflowTrackingClient,
    ],
]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[
        MlflowRunAdapter,
        MlflowScoreLogger,
        MlflowArrayLogger,
        MlflowPlotLogger,
        MlflowScoreCollectionLogger,
        MlflowArrayCollectionLogger,
        MlflowPlotCollectionLogger,
        MlflowTrackingClient,
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
        client = MlflowTrackingClient(
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
