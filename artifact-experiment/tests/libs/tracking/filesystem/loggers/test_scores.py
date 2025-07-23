import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import pytest
from artifact_experiment.libs.tracking.filesystem.adapter import FilesystemRunAdapter
from artifact_experiment.libs.tracking.filesystem.loggers.scores import FilesystemScoreLogger
from pytest_mock import MockerFixture


@pytest.fixture
def in_memory_score_store(mocker: MockerFixture) -> Dict[str, pd.DataFrame]:
    score_store = {}

    def fake_path_exists(self):
        return str(self) in score_store

    def fake_read_csv(path: Union[Path, str]):
        path_str = str(path)
        if path_str not in score_store:
            raise FileNotFoundError(f"{path_str} not found in score_store.")
        return score_store[path_str].copy()

    def fake_to_csv(self, path: Union[Path, str], index: bool = True):
        _ = index
        score_store[str(path)] = self.copy()

    mocker.patch("os.makedirs")
    mocker.patch("pandas.DataFrame.to_csv", new=fake_to_csv)
    mocker.patch("pathlib.Path.exists", new=fake_path_exists)
    mocker.patch("pandas.read_csv", side_effect=fake_read_csv)
    return score_store


@pytest.fixture
def expected_logs(request) -> Dict[str, List[float]]:
    logs = {}
    for name, val_list in request.param.items():
        logs[name] = [request.getfixturevalue(v) for v in val_list]
    return logs


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
    in_memory_score_store: dict[str, pd.DataFrame],
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
        path = os.path.join("test_root", experiment_id, run_id, "artifacts", "scores", score_name)
        df = in_memory_score_store[path]
        ls_logged_scores = df["value"].tolist()
        assert ls_logged_scores == ls_logged_scores_actual, (
            f"{score_name}: expected {ls_logged_scores_actual}, got {ls_logged_scores}"
        )
