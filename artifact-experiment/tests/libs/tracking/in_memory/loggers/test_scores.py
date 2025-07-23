from typing import Callable, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import InMemoryRunAdapter
from artifact_experiment.libs.tracking.in_memory.loggers.scores import InMemoryScoreLogger
from pytest_mock import MockerFixture


@pytest.mark.parametrize(
    "experiment_id, run_id, ls_score_names, ls_scores",
    [
        ("exp1", "run1", [], []),
        ("exp1", "run1", ["score_1"], ["score_1"]),
        (
            "exp1",
            "run1",
            ["score_1", "score_2"],
            ["score_1", "score_2"],
        ),
        (
            "exp1",
            "run1",
            ["score_1", "score_3"],
            ["score_1", "score_3"],
        ),
        (
            "exp1",
            "run1",
            ["score_1", "score_2", "score_3"],
            ["score_1", "score_2", "score_3"],
        ),
        (
            "exp1",
            "run1",
            [
                "score_1",
                "score_2",
                "score_3",
                "score_4",
                "score_5",
            ],
            [
                "score_1",
                "score_2",
                "score_3",
                "score_4",
                "score_5",
            ],
        ),
    ],
    indirect=["ls_scores"],
)
def test_log(
    mocker: MockerFixture,
    score_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryScoreLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_score_names: List[str],
    ls_scores: List[float],
):
    adapter, logger = score_logger_factory(experiment_id, run_id)
    spy = mocker.spy(adapter, "log_score")
    for name, score in zip(ls_score_names, ls_scores):
        logger.log(artifact_name=name, artifact=score)
    assert spy.call_count == len(ls_scores)
    for idx, call_args in enumerate(spy.call_args_list):
        name = ls_score_names[idx]
        score = ls_scores[idx]
        expected_path = f"{experiment_id}/{run_id}/scores/{name}/1"
        assert call_args.kwargs == {"path": expected_path, "score": score}
