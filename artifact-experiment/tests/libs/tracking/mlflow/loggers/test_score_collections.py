import os
from typing import Callable, Dict, List, Optional, Tuple

import pytest
from artifact_experiment._impl.mlflow.adapter import MlflowRunAdapter
from artifact_experiment._impl.mlflow.loggers.score_collections import (
    MlflowScoreCollectionLogger,
)
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_score_collection_names, ls_score_collections",
    [
        ("exp1", "run1", [], []),
        ("exp1", "run1", ["score_collection_1"], ["score_collection_1"]),
        (
            "exp1",
            "run1",
            ["score_collection_1", "score_collection_2"],
            ["score_collection_1", "score_collection_2"],
        ),
        (
            "exp1",
            "run1",
            ["score_collection_1", "score_collection_3"],
            ["score_collection_1", "score_collection_3"],
        ),
        (
            "exp1",
            "run1",
            ["score_collection_1", "score_collection_2", "score_collection_3"],
            ["score_collection_1", "score_collection_2", "score_collection_3"],
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
        ),
    ],
    indirect=["ls_score_collections"],
)
def test_log(
    mocker: MockerFixture,
    score_collection_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[MlflowRunAdapter, MlflowScoreCollectionLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_score_collection_names: List[str],
    ls_score_collections: List[Dict[str, float]],
):
    adapter, logger = score_collection_logger_factory(experiment_id, run_id)
    ls_logged = []
    mock_get_ls_score_history = mocker.patch.object(
        adapter, "get_ls_score_history", return_value=ls_logged
    )
    adapter_log_spy = mocker.spy(adapter, "log_score")
    for idx, (score_collection_name, score_collection) in enumerate(
        zip(ls_score_collection_names, ls_score_collections), start=1
    ):
        expected_get_call_count = len(ls_logged) + len(score_collection)
        expected_log_call_count = len(ls_logged) + len(score_collection)
        expected_score_collection_backend_path = os.path.join(
            "artifacts", "score_collections", score_collection_name
        )
        expected_step = len(ls_logged) + 1
        logger.log(artifact_name=score_collection_name, artifact=score_collection)
        assert adapter_log_spy.call_count == expected_log_call_count
        assert mock_get_ls_score_history.call_count == expected_get_call_count
        for score_name, score_value in score_collection.items():
            expected_score_backend_path = os.path.join(
                expected_score_collection_backend_path, score_name
            )
            mock_get_ls_score_history.assert_any_call(backend_path=expected_score_backend_path)
            adapter_log_spy.assert_any_call(
                backend_path=expected_score_backend_path, value=score_value, step=expected_step
            )
        for score in score_collection.values():
            ls_logged.append(score)
