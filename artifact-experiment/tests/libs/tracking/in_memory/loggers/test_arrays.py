from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pytest
from artifact_experiment._impl.backends.in_memory.adapter import InMemoryRunAdapter
from artifact_experiment._impl.backends.in_memory.loggers.arrays import InMemoryArrayLogger
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_array_names, ls_arrays, ls_step",
    [
        ("exp1", "run1", [], [], []),
        ("exp1", "run1", ["array_1"], ["array_1"], [1]),
        (
            "exp1",
            "run1",
            ["array_1", "array_2"],
            ["array_1", "array_2"],
            [1, 1],
        ),
        (
            "exp1",
            "run1",
            ["array_1", "array_3"],
            ["array_1", "array_3"],
            [1, 1],
        ),
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
            [
                "array_1",
                "array_1",
                "array_2",
                "array_2",
                "array_2",
                "array_1",
                "array_3",
            ],
            [
                "array_1",
                "array_2",
                "array_3",
                "array_1",
                "array_2",
                "array_3",
                "array_5",
            ],
            [1, 2, 1, 2, 3, 3, 1],
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
    ls_arrays: List[Array],
    ls_step: List[int],
):
    adapter, logger = array_logger_factory(experiment_id, run_id)
    spy = mocker.spy(adapter, "log_array")
    for name, array in zip(ls_array_names, ls_arrays):
        logger.log(artifact_name=name, artifact=array)
    assert spy.call_count == len(ls_arrays)
    for idx, call_args in enumerate(spy.call_args_list):
        name = ls_array_names[idx]
        array = ls_arrays[idx]
        step = ls_step[idx]
        expected_path = f"{experiment_id}/{run_id}/arrays/{name}/{step}"
        actual_path = call_args.kwargs["path"]
        assert Path(actual_path) == Path(expected_path)
        assert call_args.kwargs["array"] is array
