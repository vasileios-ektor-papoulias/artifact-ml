from typing import Callable, Dict, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryRunAdapter,
)
from artifact_experiment.libs.tracking.in_memory.loggers.score_collections import (  # noqa: E501
    InMemoryScoreCollectionLogger,
)


@pytest.mark.parametrize(
    "experiment_id, run_id, ls_score_collections",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["score_collection_1"]),
        ("exp1", "run1", ["score_collection_1", "score_collection_2"]),
        ("exp1", "run1", ["score_collection_1", "score_collection_3"]),
        ("exp1", "run1", ["score_collection_1", "score_collection_2", "score_collection_3"]),
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
        ),
    ],
    indirect=["ls_score_collections"],
)
def test_log(
    score_collection_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryScoreCollectionLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_score_collections: List[Dict[str, float]],
):
    adapter, logger = score_collection_logger_factory(experiment_id, run_id)
    for idx, score_collection in enumerate(ls_score_collections, start=1):
        logger.log(artifact_name=f"score_collection_{idx}", artifact=score_collection)
    assert adapter.n_score_collections == len(ls_score_collections)
    for idx, expected_collection in enumerate(ls_score_collections, start=1):
        key = f"{experiment_id}/{run_id}/score_collections/score_collection_{idx}/1"
        stored_collection = adapter.dict_score_collections[key]
        assert stored_collection == expected_collection
