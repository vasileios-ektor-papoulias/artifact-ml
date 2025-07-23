import os
from typing import Callable, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.filesystem.adapter import FilesystemRunAdapter
from artifact_experiment.libs.tracking.filesystem.loggers.arrays import FilesystemArrayLogger
from numpy import ndarray
from pytest_mock import MockerFixture


@pytest.fixture
def patched_incremental_generator(mocker: MockerFixture) -> List[str]:
    generated_paths: List[str] = []

    def fake_generate(dir_path: str, fmt: str) -> str:
        idx = len(generated_paths)
        path = os.path.join(dir_path, f"{idx}.{fmt}")
        generated_paths.append(path)
        return path

    mocker.patch(
        "artifact_experiment.libs.utils.incremental_path_generator.IncrementalPathGenerator.generate",
        side_effect=fake_generate,
    )

    return generated_paths


@pytest.mark.parametrize(
    "experiment_id, run_id, ls_array_names, ls_arrays",
    [
        ("exp1", "run1", [], []),
        ("exp1", "run1", ["array_1"], ["array_1"]),
        ("exp1", "run1", ["array_1", "array_2"], ["array_1", "array_2"]),
        ("exp1", "run1", ["array_1", "array_3"], ["array_1", "array_3"]),
        (
            "exp1",
            "run1",
            ["array_1", "array_2", "array_3"],
            ["array_1", "array_2", "array_3"],
        ),
        (
            "exp1",
            "run1",
            ["array_1", "array_2", "array_3", "array_4", "array_5"],
            ["array_1", "array_2", "array_3", "array_4", "array_5"],
        ),
        (
            "exp1",
            "run1",
            ["array_1", "array_2", "array_1"],
            ["array_1", "array_2", "array_3"],
        ),
        (
            "exp1",
            "run1",
            ["array_1", "array_1", "array_2", "array_2", "array_2", "array_1", "array_3"],
            ["array_1", "array_2", "array_3", "array_1", "array_2", "array_3", "array_5"],
        ),
    ],
    indirect=["ls_arrays"],
)
def test_log(
    mocker: MockerFixture,
    patched_incremental_generator: List[str],
    array_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[FilesystemRunAdapter, FilesystemArrayLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_array_names: List[str],
    ls_arrays: List[ndarray],
):
    _, logger = array_logger_factory(experiment_id, run_id)
    mock_save = mocker.patch("numpy.save")
    for name, array in zip(ls_array_names, ls_arrays):
        logger.log(artifact_name=name, artifact=array)
    assert mock_save.call_count == len(ls_arrays)
    assert len(patched_incremental_generator) == len(ls_arrays)
    for i, (name, array) in enumerate(zip(ls_array_names, ls_arrays)):
        expected_dir = os.path.join("test_root", experiment_id, run_id, "artifacts", "arrays", name)
        expected_path = os.path.join(expected_dir, f"{i}.npy")
        assert patched_incremental_generator[i] == expected_path
        mock_save.assert_any_call(file=expected_path, arr=array)
