import os
from typing import Callable, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.neptune.adapter import NeptuneRunAdapter
from artifact_experiment.libs.tracking.neptune.loggers.arrays import NeptuneArrayLogger
from numpy import ndarray
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_array_names, ls_arrays",
    [
        ("exp1", "run1", [], []),
        ("exp1", "run1", ["array_1"], ["array_1"]),
        ("exp1", "run1", ["array_1", "array_2"], ["array_1", "array_2"]),
        ("exp1", "run1", ["array_1", "array_3"], ["array_1", "array_3"]),
        ("exp1", "run1", ["array_1", "array_2", "array_3"], ["array_1", "array_2", "array_3"]),
        (
            "exp1",
            "run1",
            ["array_1", "array_2", "array_3", "array_4", "array_5"],
            ["array_1", "array_2", "array_3", "array_4", "array_5"],
        ),
    ],
    indirect=["ls_arrays"],
)
def test_log(
    mocker: MockerFixture,
    array_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[NeptuneRunAdapter, NeptuneArrayLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_array_names: List[str],
    ls_arrays: List[ndarray],
):
    adapter, logger = array_logger_factory(experiment_id, run_id)
    spy_adapter_log = mocker.spy(adapter, "log")
    for array_name, array in zip(ls_array_names, ls_arrays):
        logger.log(artifact_name=array_name, artifact=array)
    assert spy_adapter_log.call_count == len(ls_arrays)
    for idx, call_args in enumerate(spy_adapter_log.call_args_list):
        array_name = ls_array_names[idx]
        array = ls_arrays[idx]
        expected_path = os.path.join("artifacts", "arrays", array_name)
        assert call_args.kwargs == {"artifact_path": expected_path, "artifact": array}
