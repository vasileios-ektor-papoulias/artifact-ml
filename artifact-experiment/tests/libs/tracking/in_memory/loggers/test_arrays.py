from typing import Callable, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryTrackingAdapter,
)
from artifact_experiment.libs.tracking.in_memory.loggers.arrays import (
    InMemoryArrayLogger,
)
from numpy import ndarray


@pytest.mark.parametrize(
    "ls_arrays",
    [
        ([]),
        (["array_1"]),
        (["array_1", "array_2"]),
        (["array_1", "array_3"]),
        (["array_1", "array_2", "array_3"]),
        (["array_1", "array_2", "array_3", "array_4", "array_5"]),
    ],
    indirect=True,
)
def test_log(
    ls_arrays: List[ndarray],
    array_logger_factory: Callable[
        [Optional[InMemoryTrackingAdapter]], Tuple[InMemoryTrackingAdapter, InMemoryArrayLogger]
    ],
):
    adapter, logger = array_logger_factory(None)
    for idx, array in enumerate(ls_arrays, start=1):
        logger.log(artifact_name=f"array_{idx}", artifact=array)
    assert adapter.n_arrays == len(ls_arrays)
    for idx, expected_array in enumerate(ls_arrays, start=1):
        key = f"test_experiment/test_run/arrays/array_{idx}/{idx}"
        stored_array = adapter._native_run.dict_arrays[key]
        assert (stored_array == expected_array).all()
