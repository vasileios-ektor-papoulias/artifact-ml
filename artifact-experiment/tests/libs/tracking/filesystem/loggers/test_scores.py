import os
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import pytest
from artifact_experiment.libs.tracking.filesystem.adapter import FilesystemRunAdapter
from artifact_experiment.libs.tracking.filesystem.loggers.scores import FilesystemScoreLogger


@pytest.mark.parametrize(
    "experiment_id, run_id, ls_score_names, ls_scores, expected_logs",
    [
        ("exp1", "run1", [], [], {}),
        ("exp1", "run1", ["score_key_1"], ["score_1"], {"score_key_1": ["score_1"]}),
        (
            "exp1",
            "run1",
            ["score_key_1", "score_key_2"],
            ["score_1", "score_2"],
            {"score_key_1": ["score_1"], "score_key_2": ["score_2"]},
        ),
        (
            "exp1",
            "run1",
            ["score_key_1", "score_key_1"],
            ["score_1", "score_2"],
            {"score_key_1": ["score_1", "score_2"]},
        ),
        (
            "exp1",
            "run1",
            ["score_key_1", "score_key_2", "score_key_1"],
            ["score_1", "score_2", "score_3"],
            {"score_key_1": ["score_1", "score_3"], "score_key_2": ["score_2"]},
        ),
    ],
    indirect=["ls_scores", "expected_logs"],
)
def test_log(
    in_memory_df_store: dict[str, pd.DataFrame],
    score_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[FilesystemRunAdapter, FilesystemScoreLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_score_names: List[str],
    ls_scores: List[float],
    expected_logs: Dict[str, List[float]],
):
    _, logger = score_logger_factory(experiment_id, run_id)

    for score_name, score in zip(ls_score_names, ls_scores):
        logger.log(artifact_name=score_name, artifact=score)

    for score_name, ls_logged_scores_actual in expected_logs.items():
        path = os.path.join(
            "mock_home_dir", "artifact_ml", experiment_id, run_id, "artifacts", "scores", score_name
        )
        df = in_memory_df_store[path]
        ls_logged_scores = df["value"].tolist()
        assert ls_logged_scores == ls_logged_scores_actual, (
            f"{score_name}: expected {ls_logged_scores_actual}, got {ls_logged_scores}"
        )
