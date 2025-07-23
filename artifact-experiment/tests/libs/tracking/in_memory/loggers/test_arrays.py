from typing import Callable, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import InMemoryRunAdapter
from artifact_experiment.libs.tracking.in_memory.loggers.arrays import InMemoryArrayLogger
from numpy import ndarray
from pytest_mock import MockerFixture


@pytest.mark.parametrize(
    "experiment_id, run_id, ls_array_names, ls_arrays",
    [
        ("exp1", "run1", [], []),
        ("exp1", "run1", ["array_1"], ["array_1"]),
        (
            "exp1",
            "run1",
            ["array_1", "array_2"],
            ["array_1", "array_2"],
        ),
        (
            "exp1",
            "run1",
            ["array_1", "array_3"],
            ["array_1", "array_3"],
        ),
        (
            "exp1",
            "run1",
            ["array_1", "array_2", "array_3"],
            ["array_1", "array_2", "array_3"],
        ),
        (
            "exp1",
            "run1",
            [
                "array_1",
                "array_2",
                "array_3",
                "array_4",
                "array_5",
            ],
            [
                "array_1",
                "array_2",
                "array_3",
                "array_4",
                "array_5",
            ],
        ),
    ],
    indirect=["ls_arrays"],
)
def test_log(
    mocker: MockerFixture,
    array_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryArrayLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_array_names: List[str],
    ls_arrays: List[ndarray],
):
    adapter, logger = array_logger_factory(experiment_id, run_id)
    spy = mocker.spy(adapter, "log_array")
    for name, array in zip(ls_array_names, ls_arrays):
        logger.log(artifact_name=name, artifact=array)
    assert spy.call_count == len(ls_arrays)
    for idx, call_args in enumerate(spy.call_args_list):
        name = ls_array_names[idx]
        array = ls_arrays[idx]
        expected_path = f"{experiment_id}/{run_id}/arrays/{name}/1"
        assert call_args.kwargs == {"path": expected_path, "array": array}
