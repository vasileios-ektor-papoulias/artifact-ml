from typing import Callable, Optional, Tuple
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
def mock_client_factory(mocker) -> Callable[[], MagicMock]:
    def _factory() -> MagicMock:
        experiments = {}
        runs = {}
        client = mocker.MagicMock(spec=MlflowClient)

        def get_experiment_by_name(name: str):
            return experiments.get(name)

        def get_experiment(experiment_id: str):
            for exp in experiments.values():
                if exp.experiment_id == experiment_id:
                    return exp
            return None

        def create_experiment(name: str):
            if name in experiments:
                return experiments[name].experiment_id
            exp = mocker.MagicMock()
            exp.name = name
            exp.experiment_id = f"{name}_uuid"
            experiments[name] = exp
            return exp.experiment_id

        def search_runs(experiment_ids, filter_string=None):
            exp_id = experiment_ids[0]
            if not filter_string:
                return list(runs.get(exp_id, {}).values())
            if "tags.mlflow.runName" in filter_string:
                name = filter_string.split("=")[-1].strip(" '\"")
                return [run for run in runs.get(exp_id, {}).values() if run.info.run_name == name]
            return []

        def get_run(run_id: str):
            for exp_runs in runs.values():
                if run_id in exp_runs:
                    return exp_runs[run_id]
            raise ValueError("Run not found")

        def create_run(experiment_id: str, run_name: str):
            run_uuid = f"{run_name}_uuid"
            run = mocker.MagicMock(spec=Run)
            info = mocker.MagicMock()
            info.run_id = run_uuid
            info.experiment_id = experiment_id
            info.run_name = run_name
            info.status = RunStatus.to_string(RunStatus.RUNNING)
            run.info = info
            run.data = mocker.MagicMock()
            runs.setdefault(experiment_id, {})[run_uuid] = run
            return run

        client.get_experiment_by_name.side_effect = get_experiment_by_name
        client.get_experiment.side_effect = get_experiment
        client.create_experiment.side_effect = create_experiment
        client.search_runs.side_effect = search_runs
        client.get_run.side_effect = get_run
        client.create_run.side_effect = create_run

        return client

    return _factory


@pytest.fixture
def mock_experiment_factory(mocker) -> Callable[[str], MagicMock]:
    def _factory(name: str) -> MagicMock:
        experiment = mocker.MagicMock()
        experiment.name = name
        experiment.experiment_id = f"{name}_uuid"
        return experiment

    return _factory


@pytest.fixture
def mock_run_factory(mocker) -> Callable[[str, str], MagicMock]:
    def _factory(experiment_name: str, run_name: str) -> MagicMock:
        run_uuid = f"{run_name}_uuid"
        experiment_uuid = f"{experiment_name}_uuid"
        run_info = mocker.MagicMock()
        run_info.run_id = run_uuid
        run_info.experiment_id = experiment_uuid
        run_info.run_name = run_name
        run_info.status = RunStatus.to_string(RunStatus.RUNNING)
        raw_run = mocker.MagicMock(spec=Run)
        raw_run.info = run_info
        raw_run.data = mocker.MagicMock()

        return raw_run

    return _factory


@pytest.fixture
def native_run_factory(
    mock_client_factory: Callable[[], MagicMock],
) -> Callable[[Optional[str], Optional[str]], MlflowNativeRun]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> MlflowNativeRun:
        experiment_id = experiment_id or "default_experiment"
        run_id = run_id or "default_run"
        client = mock_client_factory()
        client.create_experiment(name=experiment_id)
        experiment = client.get_experiment_by_name(name=experiment_id)
        run = client.create_run(experiment_id=experiment.experiment_id, run_name=run_id)
        return MlflowNativeRun(client=client, experiment=experiment, run=run)

    return _factory


@pytest.fixture
def patch_mlflow_run_creation(
    mocker: MockerFixture,
    native_run_factory: Callable[[Optional[str], Optional[str]], MlflowNativeRun],
):
    def _patch(experiment_id: Optional[str] = None, run_id: Optional[str] = None):
        native_run = native_run_factory(experiment_id, run_id)
        mocker.patch(
            "artifact_experiment.libs.tracking.mlflow.adapter.MlflowClient",
            return_value=native_run.client,
        )
        return native_run

    return _patch


@pytest.fixture
def adapter_factory(
    native_run_factory: Callable[[Optional[str], Optional[str]], MlflowNativeRun],
) -> Callable[[Optional[str], Optional[str]], Tuple[MlflowNativeRun, MlflowRunAdapter]]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[MlflowNativeRun, MlflowRunAdapter]:
        native_run = native_run_factory(experiment_id, run_id)
        adapter = MlflowRunAdapter(native_run=native_run)
        return native_run, adapter

    return _factory


@pytest.fixture
def loggers_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MlflowNativeRun, MlflowRunAdapter],
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
