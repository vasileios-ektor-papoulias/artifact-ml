from typing import Callable, Optional, Tuple
from uuid import UUID

import numpy as np
import pytest
from artifact_core.base.artifact_dependencies import ArtifactResult
from pytest_mock import MockerFixture

from tests.base.tracking.dummy.adapter import DummyNativeRun, DummyRunAdapter

STANDARD_UUID_LENGTH = 36


@pytest.mark.parametrize(
    "experiment_id, run_id",
    [
        ("exp1", "run1"),
        ("exp1", None),
    ],
)
def test_build(
    experiment_id: str,
    run_id: Optional[str],
):
    adapter = DummyRunAdapter.build(experiment_id=experiment_id, run_id=run_id)
    assert adapter.experiment_id == experiment_id
    assert adapter.is_active is True
    if run_id is not None:
        assert adapter.run_id == run_id
    else:
        assert adapter.run_id is not None
        assert len(adapter.run_id) == STANDARD_UUID_LENGTH
        UUID(adapter.run_id)


def test_from_native_run(
    native_run_factory: Callable[[Optional[str], Optional[str]], DummyNativeRun],
):
    native_run = native_run_factory("test_exp", "test_run")
    adapter: DummyRunAdapter = DummyRunAdapter.from_native_run(native_run)
    assert adapter.experiment_id == "test_exp"
    assert adapter.run_id == "test_run"
    assert adapter.is_active is True


def test_property_delegation(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[DummyNativeRun, DummyRunAdapter]
    ],
):
    native_run, adapter = adapter_factory("test_exp", "test_run")
    assert adapter.experiment_id == native_run.experiment_id
    assert adapter.run_id == native_run.run_id
    assert adapter.is_active == native_run.is_active
    native_run.is_active = False
    assert adapter.is_active is False


def test_native_context_manager(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[DummyNativeRun, DummyRunAdapter]
    ],
):
    _, adapter = adapter_factory(None, None)
    with adapter.native() as native_run:
        assert isinstance(native_run, DummyNativeRun)
        assert native_run.experiment_id == adapter.experiment_id
        assert native_run.run_id == adapter.run_id


def test_stop_run(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[DummyNativeRun, DummyRunAdapter]
    ],
):
    native_run, adapter = adapter_factory(None, None)
    assert native_run.is_active
    assert adapter.is_active
    adapter.stop()
    assert not native_run.is_active
    assert not adapter.is_active


@pytest.mark.parametrize(
    "artifact_path, artifact",
    [
        ("/test/path/1", 2),
        ("/test/path/2", np.array([1, 2, 3])),
    ],
)
def test_log(
    mocker: MockerFixture,
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[DummyNativeRun, DummyRunAdapter]
    ],
    artifact_path: str,
    artifact: ArtifactResult,
):
    native_run, adapter = adapter_factory("test_exp", "test_run")
    native_run.log = mocker.MagicMock()
    adapter.log(artifact_path=artifact_path, artifact=artifact)
    native_run.log.assert_called_with(artifact_path=artifact_path, artifact=artifact)


@pytest.mark.parametrize(
    "path_source, dir_target",
    [
        ("/test/path", "uploads"),
        ("/data/models/model.pkl", "models"),
        ("/logs/experiment.log", "logs"),
        ("/artifacts/plot.png", "plots"),
        ("/results/summary.json", "results"),
    ],
)
def test_upload(
    mocker: MockerFixture,
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[DummyNativeRun, DummyRunAdapter]
    ],
    path_source: str,
    dir_target: str,
):
    native_run, adapter = adapter_factory("test_exp", "test_run")
    native_run.upload = mocker.MagicMock()
    adapter = DummyRunAdapter.from_native_run(native_run=native_run)
    adapter.upload(path_source=path_source, dir_target=dir_target)
    native_run.upload.assert_called_with(path_source=path_source, dir_target=dir_target)
