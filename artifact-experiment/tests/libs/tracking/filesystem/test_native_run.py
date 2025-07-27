import os
from typing import Callable, Optional

import pytest
from artifact_experiment.libs.tracking.filesystem.native_run import FilesystemRun


@pytest.mark.parametrize(
    "experiment_id, run_id",
    [("exp1", "run1")],
)
def test_init(
    native_run_factory: Callable[[Optional[str], Optional[str]], FilesystemRun],
    experiment_id: str,
    run_id: str,
):
    native_run = native_run_factory(experiment_id, run_id)
    assert native_run.experiment_id == experiment_id
    assert native_run.run_id == run_id
    assert native_run.is_active is True
    assert native_run.experiment_dir == os.path.join("mock_home_dir", "artifact_ml", experiment_id)
    assert native_run.run_dir == os.path.join("mock_home_dir", "artifact_ml", experiment_id, run_id)


@pytest.mark.parametrize(
    "experiment_id, run_id",
    [("exp1", "run1")],
)
def test_stop(
    native_run_factory: Callable[[Optional[str], Optional[str]], FilesystemRun],
    experiment_id: str,
    run_id: str,
):
    native_run = native_run_factory(experiment_id, run_id)
    assert native_run.is_active is True
    native_run.stop()
    assert native_run.is_active is False
