import os
from typing import Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock
from uuid import UUID

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
from matplotlib.figure import Figure
from numpy import ndarray
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [("exp1", "run1")],
)
def test_init(
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
    experiment_id: str,
    run_id: str,
):
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
    assert isinstance(client, NeptuneTrackingClient)
    assert client.run is adapter
    assert client.run.experiment_id == experiment_id
    assert client.run.run_id == run_id
    assert client.run.is_active is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [("exp1", "run1")],
)
def test_from_run(
    adapter_factory: Callable[[Optional[str], Optional[str]], Tuple[MagicMock, NeptuneRunAdapter]],
    experiment_id: str,
    run_id: str,
):
    _, adapter = adapter_factory(experiment_id, run_id)
    client = NeptuneTrackingClient.from_run(run=adapter)
    assert isinstance(client, NeptuneTrackingClient)
    assert client.run is adapter
    assert client.run.experiment_id == experiment_id
    assert client.run.run_id == run_id
    assert client.run.is_active is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [("exp1", "run1")],
)
def test_from_native_run(
    native_run_factory: Callable[[Optional[str], Optional[str]], MagicMock],
    experiment_id: str,
    run_id: str,
):
    native_run = native_run_factory(experiment_id, run_id)
    client = NeptuneTrackingClient.from_native_run(native_run=native_run)
    assert isinstance(client, NeptuneTrackingClient)
    assert client.run.experiment_id == experiment_id
    assert client.run.run_id == run_id
    assert client.run.is_active is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [
        ("exp1", "run1"),
        ("exp1", None),
    ],
)
def test_build(
    reset_api_token_cache,
    mock_get_env: MagicMock,
    mock_neptune_run_constructor: MagicMock,
    standard_uuid_length: int,
    experiment_id: str,
    run_id: Optional[str],
):
    client = NeptuneTrackingClient.build(experiment_id=experiment_id, run_id=run_id)
    mock_get_env.assert_called_once_with(env_var_name="NEPTUNE_API_TOKEN")
    mock_neptune_run_constructor.assert_called_once_with(
        api_token="mock-api-token",
        project=client.run.experiment_id,
        custom_run_id=client.run.run_id,
    )
    assert isinstance(client, NeptuneTrackingClient)
    assert client.run.is_active is True
    assert client.run.experiment_id == experiment_id
    if run_id is not None:
        assert client.run.run_id == run_id
    else:
        assert client.run.run_id is not None
        assert len(client.run.run_id) == standard_uuid_length
        UUID(client.run.run_id)


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_scores",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["score_1"]),
        ("exp1", "run1", ["score_1", "score_2"]),
        ("exp1", "run1", ["score_1", "score_3"]),
        ("exp1", "run1", ["score_1", "score_2", "score_3"]),
        ("exp1", "run1", ["score_1", "score_2", "score_3", "score_4", "score_5"]),
    ],
    indirect=["ls_scores"],
)
def test_log_score(
    mocker: MockerFixture,
    client_factory: Callable[
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
    ],
    experiment_id: str,
    run_id: str,
    ls_scores: List[float],
):
    (
        adapter,
        score_logger,
        _,
        _,
        _,
        _,
        _,
        client,
    ) = client_factory(experiment_id, run_id)
    spy_logger_log = mocker.spy(score_logger, "log")
    spy_adapter_log = mocker.spy(adapter, "log")
    for idx, score in enumerate(ls_scores, start=1):
        client.log_score(score=score, name=f"score_{idx}")
    assert spy_logger_log.call_count == len(ls_scores)
    for idx, score in enumerate(ls_scores, start=1):
        spy_logger_log.assert_any_call(
            artifact_name=f"score_{idx}",
            artifact=score,
        )
    assert spy_adapter_log.call_count == len(ls_scores)
    for idx, score in enumerate(ls_scores, start=1):
        expected_path = os.path.join("artifacts", "scores", f"score_{idx}")
        spy_adapter_log.assert_any_call(
            artifact_path=expected_path,
            artifact=score,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_arrays",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["array_1"]),
        ("exp1", "run1", ["array_1", "array_2"]),
        ("exp1", "run1", ["array_1", "array_3"]),
        ("exp1", "run1", ["array_1", "array_2", "array_3"]),
        ("exp1", "run1", ["array_1", "array_2", "array_3", "array_4", "array_5"]),
    ],
    indirect=["ls_arrays"],
)
def test_log_array(
    mocker: MockerFixture,
    client_factory: Callable[
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
    ],
    experiment_id: str,
    run_id: str,
    ls_arrays: List[ndarray],
):
    (
        adapter,
        _,
        array_logger,
        _,
        _,
        _,
        _,
        client,
    ) = client_factory(experiment_id, run_id)
    spy_logger_log = mocker.spy(array_logger, "log")
    spy_adapter_log = mocker.spy(adapter, "log")
    for idx, array in enumerate(ls_arrays, start=1):
        client.log_array(array=array, name=f"array_{idx}")
    assert spy_logger_log.call_count == len(ls_arrays)
    for idx, array in enumerate(ls_arrays, start=1):
        spy_logger_log.assert_any_call(
            artifact_name=f"array_{idx}",
            artifact=array,
        )
    assert spy_adapter_log.call_count == len(ls_arrays)
    for idx, array in enumerate(ls_arrays, start=1):
        expected_path = os.path.join("artifacts", "arrays", f"array_{idx}")
        spy_adapter_log.assert_any_call(
            artifact_path=expected_path,
            artifact=array,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_plots",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["plot_1"]),
        ("exp1", "run1", ["plot_1", "plot_2"]),
        ("exp1", "run1", ["plot_1", "plot_3"]),
        ("exp1", "run1", ["plot_1", "plot_2", "plot_3"]),
        ("exp1", "run1", ["plot_1", "plot_2", "plot_3", "plot_4", "plot_5"]),
    ],
    indirect=["ls_plots"],
)
def test_log_plot(
    mocker: MockerFixture,
    client_factory: Callable[
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
    ],
    experiment_id: str,
    run_id: str,
    ls_plots: List[Figure],
):
    (
        adapter,
        _,
        _,
        plot_logger,
        _,
        _,
        _,
        client,
    ) = client_factory(experiment_id, run_id)
    spy_logger_log = mocker.spy(plot_logger, "log")
    spy_adapter_log = mocker.spy(adapter, "log")
    for idx, plot in enumerate(ls_plots, start=1):
        client.log_plot(plot=plot, name=f"plot_{idx}")
    assert spy_logger_log.call_count == len(ls_plots)
    for idx, plot in enumerate(ls_plots, start=1):
        spy_logger_log.assert_any_call(
            artifact_name=f"plot_{idx}",
            artifact=plot,
        )
    assert spy_adapter_log.call_count == len(ls_plots)
    for idx, plot in enumerate(ls_plots, start=1):
        expected_path = os.path.join("artifacts", "plots", f"plot_{idx}")
        spy_adapter_log.assert_any_call(
            artifact_path=expected_path,
            artifact=plot,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_score_collections",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["score_collection_1"]),
        ("exp1", "run1", ["score_collection_1", "score_collection_2"]),
        ("exp1", "run1", ["score_collection_1", "score_collection_3"]),
        ("exp1", "run1", ["score_collection_1", "score_collection_2", "score_collection_3"]),
        (
            "exp1",
            "run1",
            [
                "score_collection_1",
                "score_collection_2",
                "score_collection_3",
                "score_collection_4",
                "score_collection_5",
            ],
        ),
    ],
    indirect=["ls_score_collections"],
)
def test_log_score_collection(
    mocker: MockerFixture,
    client_factory: Callable[
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
    ],
    experiment_id: str,
    run_id: str,
    ls_score_collections: List[Dict[str, float]],
):
    (
        adapter,
        _,
        _,
        _,
        score_collection_logger,
        _,
        _,
        client,
    ) = client_factory(experiment_id, run_id)
    spy_logger_log = mocker.spy(score_collection_logger, "log")
    spy_adapter_log = mocker.spy(adapter, "log")
    for idx, score_collection in enumerate(ls_score_collections, start=1):
        client.log_score_collection(
            score_collection=score_collection, name=f"score_collection_{idx}"
        )
    assert spy_logger_log.call_count == len(ls_score_collections)
    for idx, score_collection in enumerate(ls_score_collections, start=1):
        spy_logger_log.assert_any_call(
            artifact_name=f"score_collection_{idx}",
            artifact=score_collection,
        )
    assert spy_adapter_log.call_count == len(ls_score_collections)
    for idx, score_collection in enumerate(ls_score_collections, start=1):
        expected_path = os.path.join("artifacts", "score_collections", f"score_collection_{idx}")
        spy_adapter_log.assert_any_call(
            artifact_path=expected_path,
            artifact=score_collection,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_array_collections",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["array_collection_1"]),
        ("exp1", "run1", ["array_collection_1", "array_collection_2"]),
        ("exp1", "run1", ["array_collection_1", "array_collection_3"]),
        ("exp1", "run1", ["array_collection_1", "array_collection_2", "array_collection_3"]),
        (
            "exp1",
            "run1",
            [
                "array_collection_1",
                "array_collection_2",
                "array_collection_3",
                "array_collection_4",
                "array_collection_5",
            ],
        ),
    ],
    indirect=["ls_array_collections"],
)
def test_log_array_collection(
    mocker: MockerFixture,
    client_factory: Callable[
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
    ],
    experiment_id: str,
    run_id: str,
    ls_array_collections: List[Dict[str, ndarray]],
):
    (
        adapter,
        _,
        _,
        _,
        _,
        array_collection_logger,
        _,
        client,
    ) = client_factory(experiment_id, run_id)
    spy_logger_log = mocker.spy(array_collection_logger, "log")
    spy_adapter_log = mocker.spy(adapter, "log")
    for idx, array_collection in enumerate(ls_array_collections, start=1):
        client.log_array_collection(
            array_collection=array_collection, name=f"array_collection_{idx}"
        )
    assert spy_logger_log.call_count == len(ls_array_collections)
    for idx, array_collection in enumerate(ls_array_collections, start=1):
        spy_logger_log.assert_any_call(
            artifact_name=f"array_collection_{idx}",
            artifact=array_collection,
        )
    assert spy_adapter_log.call_count == len(ls_array_collections)
    for idx, array_collection in enumerate(ls_array_collections, start=1):
        expected_path = os.path.join("artifacts", "array_collections", f"array_collection_{idx}")
        spy_adapter_log.assert_any_call(
            artifact_path=expected_path,
            artifact=array_collection,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_plot_collections",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["plot_collection_1"]),
        ("exp1", "run1", ["plot_collection_1", "plot_collection_2"]),
        ("exp1", "run1", ["plot_collection_1", "plot_collection_3"]),
        ("exp1", "run1", ["plot_collection_1", "plot_collection_2", "plot_collection_3"]),
        (
            "exp1",
            "run1",
            [
                "plot_collection_1",
                "plot_collection_2",
                "plot_collection_3",
                "plot_collection_4",
                "plot_collection_5",
            ],
        ),
    ],
    indirect=["ls_plot_collections"],
)
def test_log_plot_collection(
    mocker: MockerFixture,
    client_factory: Callable[
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
    ],
    experiment_id: str,
    run_id: str,
    ls_plot_collections: List[Dict[str, Figure]],
):
    (
        adapter,
        _,
        _,
        _,
        _,
        _,
        plot_collection_logger,
        client,
    ) = client_factory(experiment_id, run_id)
    spy_logger_log = mocker.spy(plot_collection_logger, "log")
    spy_adapter_log = mocker.spy(adapter, "log")
    for idx, plot_collection in enumerate(ls_plot_collections, start=1):
        client.log_plot_collection(plot_collection=plot_collection, name=f"plot_collection_{idx}")
    assert spy_logger_log.call_count == len(ls_plot_collections)
    for idx, plot_collection in enumerate(ls_plot_collections, start=1):
        spy_logger_log.assert_any_call(
            artifact_name=f"plot_collection_{idx}",
            artifact=plot_collection,
        )
    assert spy_adapter_log.call_count == len(ls_plot_collections)
    for idx, plot_collection in enumerate(ls_plot_collections, start=1):
        expected_path = os.path.join("artifacts", "plot_collections", f"plot_collection_{idx}")
        spy_adapter_log.assert_any_call(
            artifact_path=expected_path,
            artifact=plot_collection,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_file_entries",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", [{"path_source": "/test/path1", "dir_target": "uploads"}]),
        (
            "exp1",
            "run1",
            [
                {"path_source": "/test/path1", "dir_target": "uploads"},
                {"path_source": "/another/path", "dir_target": "data"},
            ],
        ),
    ],
)
def test_upload(
    mocker: MockerFixture,
    client_factory: Callable[
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
    ],
    experiment_id: str,
    run_id: str,
    ls_file_entries: List[Dict[str, str]],
):
    (
        adapter,
        _,
        _,
        _,
        _,
        _,
        _,
        client,
    ) = client_factory(experiment_id, run_id)
    spy_adapter_upload = mocker.spy(adapter, "upload")
    for file_entry in ls_file_entries:
        client.upload(**file_entry)
    assert spy_adapter_upload.call_count == len(ls_file_entries)
    for entry in ls_file_entries:
        spy_adapter_upload.assert_any_call(
            path_source=entry["path_source"], dir_target=entry["dir_target"]
        )
