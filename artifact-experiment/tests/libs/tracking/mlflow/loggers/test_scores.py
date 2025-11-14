import os
from typing import Callable, List, Optional, Tuple

import pytest
from artifact_experiment._impl.mlflow.adapter import MlflowRunAdapter
from artifact_experiment._impl.mlflow.loggers.scores import MlflowScoreLogger
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_score_names, ls_scores",
    [
        ("exp1", "run1", [], []),
        ("exp1", "run1", ["score_1"], ["score_1"]),
        ("exp1", "run1", ["score_1", "score_2"], ["score_1", "score_2"]),
        ("exp1", "run1", ["score_1", "score_3"], ["score_1", "score_3"]),
        ("exp1", "run1", ["score_1", "score_2", "score_3"], ["score_1", "score_2", "score_3"]),
        (
            "exp1",
            "run1",
            ["score_1", "score_2", "score_3", "score_4", "score_5"],
            ["score_1", "score_2", "score_3", "score_4", "score_5"],
        ),
    ],
    indirect=["ls_scores"],
)
def test_log(
    mocker: MockerFixture,
    score_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[MlflowRunAdapter, MlflowScoreLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_score_names: List[str],
    ls_scores: List[float],
):
    adapter, logger = score_logger_factory(experiment_id, run_id)
    ls_logged = []
    mock_get_ls_score_history = mocker.patch.object(
        adapter, "get_ls_score_history", return_value=ls_logged
    )
    adapter_log_spy = mocker.spy(adapter, "log_score")
    for idx, (score_name, score) in enumerate(zip(ls_score_names, ls_scores), start=1):
        expected_get_call_count = idx
        expected_log_call_count = idx
        expected_backend_path = os.path.join("artifacts", "scores", score_name)
        expected_step = len(ls_logged) + 1
        logger.log(artifact_name=score_name, artifact=score)
        assert mock_get_ls_score_history.call_count == expected_get_call_count
        mock_get_ls_score_history.assert_any_call(backend_path=expected_backend_path)
        assert adapter_log_spy.call_count == expected_log_call_count
        adapter_log_spy.assert_any_call(
            backend_path=expected_backend_path, value=score, step=expected_step
        )
        ls_logged.append(score)
