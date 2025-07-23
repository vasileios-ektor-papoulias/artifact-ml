from typing import Callable, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryRunAdapter,
)
from artifact_experiment.libs.tracking.in_memory.loggers.scores import (
    InMemoryScoreLogger,
)


@pytest.mark.parametrize(
    "experiment_id, run_id, ls_scores",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["score_1"]),
        ("exp1", "run1", ["score_1", "score_2"]),
        ("exp1", "run1", ["score_1", "score_3"]),
        ("exp1", "run1", ["score_1", "score_2", "score_3"]),
        ("exp1", "run1", ["score_1", "score_2", "score_3", "score_4", "score_5"]),
    ],
    indirect=["ls_scores"],
)
def test_log(
    score_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryScoreLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_scores: List[float],
):
    adapter, logger = score_logger_factory(experiment_id, run_id)
    for idx, score in enumerate(ls_scores, start=1):
        logger.log(artifact_name=f"score_{idx}", artifact=score)
    assert adapter.n_scores == len(ls_scores)
    for idx, expected_score in enumerate(ls_scores, start=1):
        key = f"{experiment_id}/{run_id}/scores/score_{idx}/1"
        stored_score = adapter.dict_scores[key]
        assert stored_score == expected_score
