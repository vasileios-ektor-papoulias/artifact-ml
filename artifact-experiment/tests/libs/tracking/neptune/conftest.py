import os
from typing import Callable, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from artifact_experiment.libs.tracking.neptune.adapter import NeptuneRunAdapter
from artifact_experiment.libs.tracking.neptune.client import NeptuneTrackingClient
from artifact_experiment.libs.tracking.neptune.loggers.array_collections import (
    NeptuneArrayCollectionLogger,
)
from artifact_experiment.libs.tracking.neptune.loggers.arrays import NeptuneArrayLogger
from artifact_experiment.libs.tracking.neptune.loggers.plot_collections import (
    NeptunePlotCollectionLogger,
)
from artifact_experiment.libs.tracking.neptune.loggers.plots import NeptunePlotLogger
from artifact_experiment.libs.tracking.neptune.loggers.score_collections import (
    NeptuneScoreCollectionLogger,
)
from artifact_experiment.libs.tracking.neptune.loggers.scores import NeptuneScoreLogger
from pytest_mock import MockerFixture


@pytest.fixture
def native_run_factory(
    mocker: MockerFixture,
) -> Callable[[Optional[str], Optional[str]], MagicMock]:
    def _factory(experiment_id: Optional[str] = None, run_id: Optional[str] = None) -> MagicMock:
        if experiment_id is None:
            experiment_id = "default_experiment_id"
        if run_id is None:
            run_id = "default_run_id"

        channel_mocks = {}
        state_container = {"state": "running"}

        def getitem_side_effect(key: str):
            if key not in channel_mocks:
                channel = mocker.MagicMock(name=f"channel[{key}]")
                channel.upload = mocker.MagicMock(name=f"upload[{key}]")
                channel.append = mocker.MagicMock(name=f"append[{key}]")
                channel.fetch = mocker.MagicMock(name=f"fetch[{key}]")
                channel_mocks[key] = channel
            return channel_mocks[key]

        def stop_side_effect():
            state_container["state"] = "inactive"

        native_run = mocker.MagicMock(name="native_run")
        native_run.experiment_id = experiment_id
        native_run.run_id = run_id
        native_run.__getitem__.side_effect = getitem_side_effect
        native_run.stop = mocker.MagicMock(side_effect=stop_side_effect)
        native_run.fetch.side_effect = lambda: {"sys": {"state": state_container["state"]}}
        channel_mocks["sys/id"] = mocker.MagicMock()
        channel_mocks["sys/id"].fetch.return_value = run_id
        channel_mocks["sys/experiment/name"] = mocker.MagicMock()
        channel_mocks["sys/experiment/name"].fetch.return_value = experiment_id

        return native_run

    return _factory


@pytest.fixture
def patch_neptune_run_creation(
    mocker: MockerFixture,
    native_run_factory: Callable[[Optional[str], Optional[str]], MagicMock],
):
    mocker.patch(
        "artifact_experiment.libs.tracking.neptune.adapter.getpass",
        return_value="dummy-token",
    )

    def init_run_side_effect(api_token: str, project: str, custom_run_id: Optional[str] = None):
        _ = api_token
        native_run = native_run_factory(project, custom_run_id)
        native_run.experiment_id = project
        native_run.run_id = custom_run_id
        return native_run

    mocker.patch("neptune.init_run", side_effect=init_run_side_effect)


@pytest.fixture
def adapter_factory(
    native_run_factory: Callable[[Optional[str], Optional[str]], MagicMock],
) -> Callable[[Optional[str], Optional[str]], Tuple[MagicMock, NeptuneRunAdapter]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[MagicMock, NeptuneRunAdapter]:
        if experiment_id is None:
            experiment_id = "test_experiment"
        if run_id is None:
            run_id = "test_run"

        native_run = native_run_factory(experiment_id, run_id)
        adapter = NeptuneRunAdapter(native_run=native_run)
        return native_run, adapter

    return _factory


@pytest.fixture
def loggers_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MagicMock, NeptuneRunAdapter],
    ],
) -> Callable[
    [Optional[str], Optional[str]],
    Tuple[
        NeptuneRunAdapter,
        NeptuneScoreLogger,
        NeptuneArrayLogger,
        NeptunePlotLogger,
        NeptuneScoreCollectionLogger,
        NeptuneArrayCollectionLogger,
        NeptunePlotCollectionLogger,
    ],
]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[
        NeptuneRunAdapter,
        NeptuneScoreLogger,
        NeptuneArrayLogger,
        NeptunePlotLogger,
        NeptuneScoreCollectionLogger,
        NeptuneArrayCollectionLogger,
        NeptunePlotCollectionLogger,
    ]:
        _, adapter = adapter_factory(experiment_id, run_id)
        score_logger = NeptuneScoreLogger(run=adapter)
        array_logger = NeptuneArrayLogger(run=adapter)
        plot_logger = NeptunePlotLogger(run=adapter)
        score_collection_logger = NeptuneScoreCollectionLogger(run=adapter)
        array_collection_logger = NeptuneArrayCollectionLogger(run=adapter)
        plot_collection_logger = NeptunePlotCollectionLogger(run=adapter)
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
            NeptuneRunAdapter,
            NeptuneScoreLogger,
            NeptuneArrayLogger,
            NeptunePlotLogger,
            NeptuneScoreCollectionLogger,
            NeptuneArrayCollectionLogger,
            NeptunePlotCollectionLogger,
        ],
    ],
) -> Callable[
    [Optional[str], Optional[str]],
    Tuple[
        NeptuneRunAdapter,
        NeptuneScoreLogger,
        NeptuneArrayLogger,
        NeptunePlotLogger,
        NeptuneScoreCollectionLogger,
        NeptuneArrayCollectionLogger,
        NeptunePlotCollectionLogger,
        NeptuneTrackingClient,
    ],
]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[
        NeptuneRunAdapter,
        NeptuneScoreLogger,
        NeptuneArrayLogger,
        NeptunePlotLogger,
        NeptuneScoreCollectionLogger,
        NeptuneArrayCollectionLogger,
        NeptunePlotCollectionLogger,
        NeptuneTrackingClient,
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
        client = NeptuneTrackingClient(
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


@pytest.fixture
def get_absolute_log_path() -> Callable[[str], str]:
    def _prepend(path: str) -> str:
        return os.path.join("artifact_ml", path.lstrip("/")).replace("\\", "/")

    return _prepend
