import os
from typing import Callable, Dict, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.neptune.adapter import NeptuneRunAdapter
from artifact_experiment.libs.tracking.neptune.loggers.array_collections import (
    NeptuneArrayCollectionLogger,
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
        [Optional[str], Optional[str]], Tuple[NeptuneRunAdapter, NeptuneArrayCollectionLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_array_collection_names: List[str],
    ls_array_collections: List[Dict[str, ndarray]],
):
    adapter, logger = array_collection_logger_factory(experiment_id, run_id)
    spy_adapter_log = mocker.spy(adapter, "log")
    for array_collection_name, array_collection in zip(
        ls_array_collection_names, ls_array_collections
    ):
        logger.log(artifact_name=array_collection_name, artifact=array_collection)
    assert spy_adapter_log.call_count == len(ls_array_collections)
    for idx, call_args in enumerate(spy_adapter_log.call_args_list):
        array_collection_name = ls_array_collection_names[idx]
        array_collection = ls_array_collections[idx]
        expected_path = os.path.join("artifacts", "array_collections", array_collection_name)
        assert call_args.kwargs == {"artifact_path": expected_path, "artifact": array_collection}
