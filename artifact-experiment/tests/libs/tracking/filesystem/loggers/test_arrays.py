import os
from typing import Callable, List, Optional, Tuple

import pytest
from artifact_experiment._impl.filesystem.adapter import FilesystemRunAdapter
from artifact_experiment._impl.filesystem.loggers.arrays import FilesystemArrayLogger
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_array_names, ls_arrays, ls_step",
    [
        ("exp1", "run1", [], [], []),
        ("exp1", "run1", ["array_1"], ["array_1"], [1]),
        ("exp1", "run1", ["array_1", "array_2"], ["array_1", "array_2"], [1, 1]),
        ("exp1", "run1", ["array_1", "array_3"], ["array_1", "array_3"], [1, 1]),
        (
            "exp1",
            "run1",
            ["array_1", "array_2", "array_3"],
            ["array_1", "array_2", "array_3"],
            [1, 1, 1],
        ),
        (
            "exp1",
            "run1",
            ["array_1", "array_2", "array_3", "array_4", "array_5"],
            ["array_1", "array_2", "array_3", "array_4", "array_5"],
            [1, 1, 1, 1, 1],
        ),
        (
            "exp1",
            "run1",
            ["array_1", "array_2", "array_1"],
            ["array_1", "array_2", "array_3"],
            [1, 1, 2],
        ),
        (
            "exp1",
            "run1",
            ["array_1", "array_1", "array_2", "array_2", "array_2", "array_1", "array_3"],
            ["array_1", "array_2", "array_3", "array_1", "array_2", "array_3", "array_5"],
            [1, 2, 1, 2, 3, 3, 1],
        ),
    ],
    indirect=["ls_arrays"],
)
def test_log(
    mocker: MockerFixture,
    mock_incremental_path_generator: List[str],
    array_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[FilesystemRunAdapter, FilesystemArrayLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_array_names: List[str],
    ls_arrays: List[Array],
    ls_step: List[int],
):
    _, logger = array_logger_factory(experiment_id, run_id)
    mock_save = mocker.patch("numpy.save")
    for name, array in zip(ls_array_names, ls_arrays):
        logger.log(artifact_name=name, artifact=array)
    assert mock_save.call_count == len(ls_arrays)
    assert len(mock_incremental_path_generator) == len(ls_arrays)
    for i, (name, array, step) in enumerate(zip(ls_array_names, ls_arrays, ls_step)):
        expected_dir = os.path.join(
            "mock_home_dir", "artifact_ml", experiment_id, run_id, "artifacts", "arrays", name
        )
        expected_path = os.path.join(expected_dir, f"{step}.npy")
        assert mock_incremental_path_generator[i] == expected_path
        mock_save.assert_any_call(file=expected_path, arr=array)
    for call in mock_save.call_args_list:
        print(call)
