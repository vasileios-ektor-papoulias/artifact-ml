from typing import Callable, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryRunAdapter,
)
from artifact_experiment.libs.tracking.in_memory.loggers.arrays import (
    InMemoryArrayLogger,
)
from numpy import ndarray


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
def test_log(
    array_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryArrayLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_arrays: List[ndarray],
):
    adapter, logger = array_logger_factory(experiment_id, run_id)
    for idx, array in enumerate(ls_arrays, start=1):
        logger.log(artifact_name=f"array_{idx}", artifact=array)
    assert adapter.n_arrays == len(ls_arrays)
    for idx, expected_array in enumerate(ls_arrays, start=1):
        key = f"{experiment_id}/{run_id}/arrays/array_{idx}/1"
        stored_array = adapter.dict_arrays[key]
        assert (stored_array == expected_array).all()
