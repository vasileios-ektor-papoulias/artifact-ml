from typing import Callable, Dict, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryRunAdapter,
)
from artifact_experiment.libs.tracking.in_memory.loggers.score_collections import (  # noqa: E501
    InMemoryScoreCollectionLogger,
)


@pytest.mark.parametrize(
    "ls_score_collections",
    [
        ([]),
        (["score_collection_1"]),
        (["score_collection_1", "score_collection_2"]),
        (["score_collection_1", "score_collection_3"]),
        (["score_collection_1", "score_collection_2", "score_collection_3"]),
        (
            [
                "score_collection_1",
                "score_collection_2",
                "score_collection_3",
                "score_collection_4",
                "score_collection_5",
            ]
        ),
    ],
    indirect=True,
)
def test_log(
    ls_score_collections: List[Dict[str, float]],
    score_collection_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryScoreCollectionLogger]
    ],
):
    adapter, logger = score_collection_logger_factory("test_experiment", "test_run")
    for idx, score_collection in enumerate(ls_score_collections, start=1):
        logger.log(artifact_name=f"score_collection_{idx}", artifact=score_collection)
    assert adapter.n_score_collections == len(ls_score_collections)
    for idx, expected_collection in enumerate(ls_score_collections, start=1):
        key = f"test_experiment/test_run/score_collections/score_collection_{idx}/1"
        stored_collection = adapter.dict_score_collections[key]
        assert stored_collection == expected_collection
