import os
from typing import Callable, Optional, Tuple
from uuid import UUID

import pytest
from artifact_experiment._impl.filesystem.adapter import FilesystemRunAdapter
from artifact_experiment._impl.filesystem.native_run import FilesystemRun
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [
        ("exp1", "run1"),
        ("exp1", None),
    ],
)
def test_build(
    patch_filesystem_run_creation,
    standard_uuid_length: int,
    experiment_id: str,
    run_id: Optional[str],
):
    adapter = FilesystemRunAdapter.build(experiment_id=experiment_id, run_id=run_id)
    assert adapter.is_active is True
    assert adapter.experiment_id == experiment_id
    if run_id is not None:
        assert adapter.run_id == run_id
    else:
        assert adapter.run_id is not None
        assert len(adapter.run_id) == standard_uuid_length
        UUID(adapter.run_id)


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [
        ("exp1", "run1"),
        ("exp1", None),
    ],
)
def test_from_native_run(
    native_run_factory: Callable[[Optional[str], Optional[str]], FilesystemRun],
    experiment_id: str,
    run_id: Optional[str],
):
    native_run = native_run_factory(experiment_id, run_id)
    adapter = FilesystemRunAdapter.from_native_run(native_run=native_run)
    assert adapter.is_active is True
    assert adapter.experiment_id == experiment_id
    assert adapter.experiment_id == native_run.experiment_id
    assert adapter.run_id == native_run.run_id
    if run_id is not None:
        assert adapter.run_id == run_id
        assert native_run.run_id == run_id


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [
        ("exp1", "run1"),
        ("exp1", None),
    ],
)
def test_native_context_manager(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[FilesystemRun, FilesystemRunAdapter]
    ],
    experiment_id: str,
    run_id: Optional[str],
):
    native_run, adapter = adapter_factory(experiment_id, run_id)
    with adapter.native() as ctx_native_run:
        assert ctx_native_run is native_run


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [
        ("exp1", "run1"),
        ("exp1", None),
    ],
)
def test_stop_run(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[FilesystemRun, FilesystemRunAdapter]
    ],
    experiment_id: str,
    run_id: Optional[str],
):
    _, adapter = adapter_factory(experiment_id, run_id)
    assert adapter.is_active
    adapter.stop()
    assert not adapter.is_active


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, path_source, dir_target",
    [
        ("exp1", "run1", "/test/path", "uploads"),
        ("exp1", None, "/data/models/model.pkl", "models"),
        ("exp1", "run1", "/logs/experiment.log", "logs"),
        ("exp1", None, "/artifacts/plot.png", "plots"),
        ("exp1", "run1", "/results/summary.json", "results"),
    ],
)
def test_upload(
    mocker: MockerFixture,
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[FilesystemRun, FilesystemRunAdapter]
    ],
    experiment_id: str,
    run_id: Optional[str],
    path_source: str,
    dir_target: str,
):
    native_run, adapter = adapter_factory(experiment_id, run_id)
    mock_copy = mocker.patch("shutil.copy2")
    adapter.upload(path_source=path_source, dir_target=dir_target)
    expected_target_dir = os.path.join(native_run.run_dir, dir_target)
    mock_copy.assert_called_once_with(path_source, expected_target_dir)
