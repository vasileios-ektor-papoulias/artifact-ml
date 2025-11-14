from typing import Callable, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from artifact_experiment._impl.mlflow.adapter import (
    MlflowNativeRun,
    MlflowRunAdapter,
)
from artifact_experiment._impl.mlflow.client import MlflowTrackingClient
from artifact_experiment._impl.mlflow.loggers.array_collections import (
    MlflowArrayCollectionLogger,
)
from artifact_experiment._impl.mlflow.loggers.arrays import MlflowArrayLogger
from artifact_experiment._impl.mlflow.loggers.plot_collections import (
    MlflowPlotCollectionLogger,
)
from artifact_experiment._impl.mlflow.loggers.plots import MlflowPlotLogger
from artifact_experiment._impl.mlflow.loggers.score_collections import (
    MlflowScoreCollectionLogger,
)
from artifact_experiment._impl.mlflow.loggers.scores import MlflowScoreLogger
from mlflow.entities import Run, RunStatus
from mlflow.tracking import MlflowClient
from pytest_mock import MockerFixture


@pytest.fixture
def mock_experiment_factory(mocker: MockerFixture) -> Callable[[str], MagicMock]:
    def _factory(name: str) -> MagicMock:
        mock_experiment = mocker.MagicMock()
        mock_experiment.name = name
        mock_experiment.experiment_id = f"{name}_uuid"
        return mock_experiment

    return _factory


@pytest.fixture
def mock_run_factory(mocker: MockerFixture) -> Callable[[str, str], MagicMock]:
    def _factory(experiment_name: str, run_name: str) -> MagicMock:
        run_uuid = f"{run_name}_uuid"
        experiment_uuid = f"{experiment_name}_uuid"
        mock_run_info = mocker.MagicMock()
        mock_run_info.run_id = run_uuid
        mock_run_info.experiment_id = experiment_uuid
        mock_run_info.run_name = run_name
        mock_run_info.status = RunStatus.to_string(RunStatus.RUNNING)
        mock_run = mocker.MagicMock(spec=Run)
        mock_run.info = mock_run_info

        return mock_run

    return _factory


@pytest.fixture
def mock_client(
    mocker: MockerFixture,
    mock_experiment_factory: Callable[[str], MagicMock],
    mock_run_factory: Callable[[str, str], MagicMock],
) -> MagicMock:
    experiments = {}
    runs = {}

    def get_experiment_by_name(name: str):
        return experiments.get(name)

    def get_experiment(experiment_id: str):
        return next(
            (exp for exp in experiments.values() if exp.experiment_id == experiment_id),
            None,
        )

    def create_experiment(name: str):
        if name in experiments:
            return experiments[name].experiment_id
        exp = mock_experiment_factory(name)
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
        exp = get_experiment(experiment_id)
        if exp is None:
            raise ValueError(f"Experiment ID {experiment_id} not found")

        run = mock_run_factory(exp.name, run_name)
        runs.setdefault(experiment_id, {})[run.info.run_id] = run
        return run

    def set_terminated(run_id: str):
        for exp_runs in runs.values():
            if run_id in exp_runs:
                exp_runs[run_id].info.status = RunStatus.to_string(RunStatus.FINISHED)
                return
        raise ValueError(f"Run ID {run_id} not found to terminate")

    mock_client = mocker.MagicMock(spec=MlflowClient)
    mock_client.get_experiment_by_name.side_effect = get_experiment_by_name
    mock_client.get_experiment.side_effect = get_experiment
    mock_client.create_experiment.side_effect = create_experiment
    mock_client.search_runs.side_effect = search_runs
    mock_client.get_run.side_effect = get_run
    mock_client.create_run.side_effect = create_run
    mock_client.set_terminated.side_effect = set_terminated

    return mock_client


@pytest.fixture
def mock_mlflow_client_constructor(mocker: MockerFixture, mock_client: MagicMock) -> MagicMock:
    mock_client_constructor = mocker.patch(
        "artifact_experiment.libs.tracking.mlflow.adapter.MlflowClient",
        return_value=mock_client,
    )
    return mock_client_constructor


@pytest.fixture
def mock_get_env(mocker: MockerFixture) -> MagicMock:
    get_env_mock = mocker.patch(
        "artifact_experiment.libs.utils.environment_variable_reader.EnvironmentVariableReader.get",
        return_value="mock-tracking_uri",
    )
    return get_env_mock


@pytest.fixture
def native_run_factory(
    mock_get_env: MagicMock,
    mock_mlflow_client_constructor: MagicMock,
    mock_client: MagicMock,
) -> Callable[
    [Optional[str], Optional[str]], Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun]
]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun]:
        experiment_id = experiment_id or "default_experiment"
        run_id = run_id or "default_run"
        mock_client.create_experiment(name=experiment_id)
        mock_experiment = mock_client.get_experiment_by_name(name=experiment_id)
        mock_run = mock_client.create_run(
            experiment_id=mock_experiment.experiment_id, run_name=run_id
        )
        native_run = MlflowNativeRun(client=mock_client, experiment=mock_experiment, run=mock_run)
        return mock_client, mock_experiment, mock_run, native_run

    return _factory


@pytest.fixture(autouse=True)
def reset_tracking_uri_cache():
    MlflowRunAdapter._tracking_uri = None
    yield
    MlflowRunAdapter._tracking_uri = None


@pytest.fixture
def adapter_factory(
    native_run_factory: Callable[
        [Optional[str], Optional[str]], Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun]
    ],
) -> Callable[
    [Optional[str], Optional[str]],
    Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun, MlflowRunAdapter],
]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun, MlflowRunAdapter]:
        mock_client, mock_experiment, mock_run, native_run = native_run_factory(
            experiment_id, run_id
        )
        adapter = MlflowRunAdapter(native_run=native_run)
        return mock_client, mock_experiment, mock_run, native_run, adapter

    return _factory


@pytest.fixture
def loggers_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun, MlflowRunAdapter],
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
        _, _, _, _, adapter = adapter_factory(experiment_id, run_id)
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
