from typing import Callable, Dict, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import InMemoryRunAdapter
from artifact_experiment.libs.tracking.in_memory.loggers.score_collections import (
    InMemoryScoreCollectionLogger,
)
from pytest_mock import MockerFixture


@pytest.mark.parametrize(
    "experiment_id, run_id, ls_score_collection_names, ls_score_collections, ls_step",
    [
        ("exp1", "run1", [], [], []),
        ("exp1", "run1", ["score_collection_1"], ["score_collection_1"], [1]),
        (
            "exp1",
            "run1",
            ["score_collection_1", "score_collection_2"],
            ["score_collection_1", "score_collection_2"],
            [1, 1],
        ),
        (
            "exp1",
            "run1",
            ["score_collection_1", "score_collection_3"],
            ["score_collection_1", "score_collection_3"],
            [1, 1],
        ),
        (
            "exp1",
            "run1",
            ["score_collection_1", "score_collection_2", "score_collection_3"],
            ["score_collection_1", "score_collection_2", "score_collection_3"],
            [1, 1, 1],
        ),
        (
            "exp1",
            "run1",
            [
                "score_collection_1",
                "score_collection_2",
                "score_collection_3",
                "score_collection_4",
                "score_collection_5",
            ],
            [
                "score_collection_1",
                "score_collection_2",
                "score_collection_3",
                "score_collection_4",
                "score_collection_5",
            ],
            [1, 1, 1, 1, 1],
        ),
        (
            "exp1",
            "run1",
            ["score_collection_1", "score_collection_2", "score_collection_1"],
            ["score_collection_1", "score_collection_2", "score_collection_3"],
            [1, 1, 2],
        ),
        (
            "exp1",
            "run1",
            [
                "score_collection_1",
                "score_collection_1",
                "score_collection_2",
                "score_collection_2",
                "score_collection_2",
                "score_collection_1",
                "score_collection_3",
            ],
            [
                "score_collection_1",
                "score_collection_2",
                "score_collection_3",
                "score_collection_1",
                "score_collection_2",
                "score_collection_3",
                "score_collection_5",
            ],
            [1, 2, 1, 2, 3, 3, 1],
        ),
    ],
    indirect=["ls_score_collections"],
)
def test_log(
    mocker: MockerFixture,
    score_collection_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryScoreCollectionLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_score_collection_names: List[str],
    ls_score_collections: List[Dict[str, float]],
    ls_step: List[int],
):
    adapter, logger = score_collection_logger_factory(experiment_id, run_id)
    spy = mocker.spy(adapter, "log_score_collection")
    for name, score_collection in zip(ls_score_collection_names, ls_score_collections):
        logger.log(artifact_name=name, artifact=score_collection)
    assert spy.call_count == len(ls_score_collections)
    for idx, call_args in enumerate(spy.call_args_list):
        name = ls_score_collection_names[idx]
        score_collection = ls_score_collections[idx]
        step = ls_step[idx]
        expected_path = f"{experiment_id}/{run_id}/score_collections/{name}/{step}"
        assert call_args.kwargs == {"path": expected_path, "score_collection": score_collection}
