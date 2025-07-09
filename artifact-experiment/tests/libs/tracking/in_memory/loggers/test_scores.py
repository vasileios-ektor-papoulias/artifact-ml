from typing import Callable, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryRunAdapter,
)
from artifact_experiment.libs.tracking.in_memory.loggers.scores import (
    InMemoryScoreLogger,
)


@pytest.mark.parametrize(
    "ls_scores",
    [
        ([]),
        (["score_1"]),
        (["score_1", "score_2"]),
        (["score_1", "score_3"]),
        (["score_1", "score_2", "score_3"]),
        (["score_1", "score_2", "score_3", "score_4", "score_5"]),
    ],
    indirect=True,
)
def test_log(
    ls_scores: List[float],
    score_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryScoreLogger]
    ],
):
    adapter, logger = score_logger_factory("test_experiment", "test_run")
    for idx, score in enumerate(ls_scores, start=1):
        logger.log(artifact_name=f"score_{idx}", artifact=score)
    assert adapter.n_scores == len(ls_scores)
    for idx, expected_score in enumerate(ls_scores, start=1):
        key = f"test_experiment/test_run/scores/score_{idx}/1"
        stored_score = adapter.dict_scores[key]
        assert stored_score == expected_score
