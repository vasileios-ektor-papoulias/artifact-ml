from typing import Callable, Dict, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import InMemoryRunAdapter
from artifact_experiment.libs.tracking.in_memory.loggers.array_collections import (
    InMemoryArrayCollectionLogger,
)
from numpy import ndarray
from pytest_mock import MockerFixture


@pytest.mark.parametrize(
    "experiment_id, run_id, ls_array_collection_names, ls_array_collections",
    [
        ("exp1", "run1", [], []),
        ("exp1", "run1", ["array_collection_1"], ["array_collection_1"]),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_2"],
            ["array_collection_1", "array_collection_2"],
        ),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_3"],
            ["array_collection_1", "array_collection_3"],
        ),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_2", "array_collection_3"],
            ["array_collection_1", "array_collection_2", "array_collection_3"],
        ),
        (
            "exp1",
            "run1",
            [
                "array_collection_1",
                "array_collection_2",
                "array_collection_3",
                "array_collection_4",
                "array_collection_5",
            ],
            [
                "array_collection_1",
                "array_collection_2",
                "array_collection_3",
                "array_collection_4",
                "array_collection_5",
            ],
        ),
    ],
    indirect=["ls_array_collections"],
)
def test_log(
    mocker: MockerFixture,
    array_collection_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryArrayCollectionLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_array_collection_names: List[str],
    ls_array_collections: List[Dict[str, ndarray]],
):
    adapter, logger = array_collection_logger_factory(experiment_id, run_id)
    spy = mocker.spy(adapter, "log_array_collection")
    for name, collection in zip(ls_array_collection_names, ls_array_collections):
        logger.log(artifact_name=name, artifact=collection)
    assert spy.call_count == len(ls_array_collections)
    for idx, call_args in enumerate(spy.call_args_list):
        name = ls_array_collection_names[idx]
        array_collection = ls_array_collections[idx]
        expected_path = f"{experiment_id}/{run_id}/array_collections/{name}/1"
        assert call_args.kwargs == {"path": expected_path, "array_collection": array_collection}
