from typing import Callable, Dict, List, Optional, Tuple
from uuid import UUID

import pytest
from pytest_mock import MockerFixture

from tests.base.tracking.dummy.adapter import DummyNativeRun, DummyRunAdapter
from tests.base.tracking.dummy.client import DummyTrackingClient
from tests.base.tracking.dummy.logger import DummyArtifactLogger


def _get_name(key: str, idx: int) -> str:
    return f"{key}_{idx}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [("exp1", "run1")],
)
def test_init(
    logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[DummyRunAdapter, DummyArtifactLogger]
    ],
    experiment_id: str,
    run_id: str,
):
    adapter, logger = logger_factory(experiment_id, run_id)
    client = DummyTrackingClient(
        run=adapter,
        score_logger=logger,
        array_logger=logger,
        plot_logger=logger,
        score_collection_logger=logger,
        array_collection_logger=logger,
        plot_collection_logger=logger,
    )
    assert isinstance(client, DummyTrackingClient)
    assert client.run == adapter
    assert client.run.experiment_id == experiment_id
    assert client.run.run_id == run_id
    assert client.run.is_active is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [("exp1", "run1")],
)
def test_from_run(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[DummyNativeRun, DummyRunAdapter]
    ],
    experiment_id: str,
    run_id: str,
):
    _, adapter = adapter_factory(experiment_id, run_id)
    client = DummyTrackingClient.from_run(run=adapter)
    assert isinstance(client, DummyTrackingClient)
    assert client.run == adapter
    assert client.run.experiment_id == experiment_id
    assert client.run.run_id == run_id
    assert client.run.is_active is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [("exp1", "run1")],
)
def test_from_native_run(
    native_run_factory: Callable[[Optional[str], Optional[str]], DummyNativeRun],
    experiment_id: str,
    run_id: str,
):
    native_run = native_run_factory(experiment_id, run_id)
    client = DummyTrackingClient.from_native_run(native_run=native_run)
    assert isinstance(client, DummyTrackingClient)
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
    standard_uuid_length: int,
    experiment_id: str,
    run_id: Optional[str],
):
    client = DummyTrackingClient.build(experiment_id=experiment_id, run_id=run_id)
    assert isinstance(client, DummyTrackingClient)
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
        Tuple[DummyRunAdapter, DummyArtifactLogger, DummyTrackingClient],
    ],
    experiment_id: str,
    run_id: str,
    ls_scores: List[float],
):
    _, logger, client = client_factory(experiment_id, run_id)
    logger.log = mocker.MagicMock()
    for idx, score in enumerate(ls_scores):
        name = _get_name(key="score", idx=idx)
        client.log_score(score=score, name=name)
    assert logger.log.call_count == len(ls_scores)
    for idx, call_args in enumerate(logger.log.call_args_list):
        score = ls_scores[idx]
        name = _get_name(key="score", idx=idx)
        assert call_args.kwargs == {"artifact_name": name, "artifact": score}


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
        Tuple[DummyRunAdapter, DummyArtifactLogger, DummyTrackingClient],
    ],
    experiment_id: str,
    run_id: str,
    ls_arrays: List[Array],
):
    _, logger, client = client_factory(experiment_id, run_id)
    logger.log = mocker.MagicMock()
    for idx, array in enumerate(ls_arrays):
        name = _get_name("array", idx)
        client.log_array(array=array, name=name)
    assert logger.log.call_count == len(ls_arrays)
    for idx, call_args in enumerate(logger.log.call_args_list):
        array = ls_arrays[idx]
        name = _get_name("array", idx)
        assert call_args.kwargs == {"artifact_name": name, "artifact": array}


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
        Tuple[DummyRunAdapter, DummyArtifactLogger, DummyTrackingClient],
    ],
    experiment_id: str,
    run_id: str,
    ls_plots: List[Figure],
):
    _, logger, client = client_factory(experiment_id, run_id)
    logger.log = mocker.MagicMock()

    for idx, plot in enumerate(ls_plots):
        name = _get_name("plot", idx)
        client.log_plot(plot=plot, name=name)
    assert logger.log.call_count == len(ls_plots)
    for idx, call_args in enumerate(logger.log.call_args_list):
        plot = ls_plots[idx]
        name = _get_name("plot", idx)
        assert call_args.kwargs == {"artifact_name": name, "artifact": plot}


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
        Tuple[DummyRunAdapter, DummyArtifactLogger, DummyTrackingClient],
    ],
    experiment_id: str,
    run_id: str,
    ls_score_collections: List[Dict[str, float]],
):
    _, logger, client = client_factory(experiment_id, run_id)
    logger.log = mocker.MagicMock()
    for idx, score_collection in enumerate(ls_score_collections):
        name = _get_name("score_collection", idx)
        client.log_score_collection(score_collection=score_collection, name=name)
    assert logger.log.call_count == len(ls_score_collections)
    for idx, call_args in enumerate(logger.log.call_args_list):
        collection = ls_score_collections[idx]
        name = _get_name("score_collection", idx)
        assert call_args.kwargs == {"artifact_name": name, "artifact": collection}


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
        Tuple[DummyRunAdapter, DummyArtifactLogger, DummyTrackingClient],
    ],
    experiment_id: str,
    run_id: str,
    ls_array_collections: List[Dict[str, Array]],
):
    _, logger, client = client_factory(experiment_id, run_id)
    logger.log = mocker.MagicMock()
    for idx, array_collection in enumerate(ls_array_collections):
        name = _get_name("array_collection", idx)
        client.log_array_collection(array_collection=array_collection, name=name)
    assert logger.log.call_count == len(ls_array_collections)
    for idx, call_args in enumerate(logger.log.call_args_list):
        collection = ls_array_collections[idx]
        name = _get_name("array_collection", idx)
        assert call_args.kwargs == {"artifact_name": name, "artifact": collection}


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
    experiment_id: str,
    run_id: str,
    ls_plot_collections: List[Dict[str, Figure]],
    client_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[DummyRunAdapter, DummyArtifactLogger, DummyTrackingClient],
    ],
):
    _, logger, client = client_factory(experiment_id, run_id)
    logger.log = mocker.MagicMock()
    for idx, plot_collection in enumerate(ls_plot_collections):
        name = _get_name("plot_collection", idx)
        client.log_plot_collection(plot_collection=plot_collection, name=name)
    assert logger.log.call_count == len(ls_plot_collections)
    for idx, call_args in enumerate(logger.log.call_args_list):
        collection = ls_plot_collections[idx]
        name = _get_name("plot_collection", idx)
        assert call_args.kwargs == {"artifact_name": name, "artifact": collection}


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, path_source, dir_target",
    [
        ("exp1", "run1", "/test/path", "uploads"),
        ("exp1", "run1", "/data/file.pkl", "models"),
        ("exp1", "run1", "/logs/run.log", "logs"),
    ],
)
def test_upload(
    mocker: MockerFixture,
    client_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[DummyRunAdapter, DummyArtifactLogger, DummyTrackingClient],
    ],
    experiment_id: str,
    run_id: str,
    path_source: str,
    dir_target: str,
):
    adapter, _, client = client_factory(experiment_id, run_id)
    adapter.upload = mocker.MagicMock()
    client.log_file(path_source=path_source, dir_target=dir_target)
    adapter.upload.assert_called_once_with(path_source=path_source, dir_target=dir_target)
