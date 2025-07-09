from typing import Callable, Dict, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryRunAdapter,
)
from artifact_experiment.libs.tracking.in_memory.loggers.array_collections import (
    InMemoryArrayCollectionLogger,
)
from numpy import ndarray


@pytest.mark.parametrize(
    "ls_array_collections",
    [
        ([]),
        (["array_collection_1"]),
        (["array_collection_1", "array_collection_2"]),
        (["array_collection_1", "array_collection_3"]),
        (["array_collection_1", "array_collection_2", "array_collection_3"]),
        (
            [
                "array_collection_1",
                "array_collection_2",
                "array_collection_3",
                "array_collection_4",
                "array_collection_5",
            ]
        ),
    ],
    indirect=True,
)
def test_log(
    ls_array_collections: List[Dict[str, ndarray]],
    array_collection_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryArrayCollectionLogger]
    ],
):
    adapter, logger = array_collection_logger_factory("test_experiment", "test_run")
    for idx, array_collection in enumerate(ls_array_collections, start=1):
        logger.log(artifact_name=f"array_collection_{idx}", artifact=array_collection)
        print(adapter.dict_array_collections)
    assert adapter.n_array_collections == len(ls_array_collections)
    for idx, expected_collection in enumerate(ls_array_collections, start=1):
        key = f"test_experiment/test_run/array_collections/array_collection_{idx}/1"
        stored_collection = adapter._native_run.dict_array_collections[key]
        assert stored_collection.keys() == expected_collection.keys()
        for arr_name, expected_array in expected_collection.items():
            assert (stored_collection[arr_name] == expected_array).all()
