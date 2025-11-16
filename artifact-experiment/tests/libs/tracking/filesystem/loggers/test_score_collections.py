import os
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import pytest
from artifact_experiment._impl.backends.filesystem.adapter import FilesystemRunAdapter
from artifact_experiment._impl.backends.filesystem.loggers.score_collections import (
    FilesystemScoreCollectionLogger,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_score_collection_names, ls_score_collections, expected_logs",
    [
        ("exp1", "run1", [], [], {}),
        (
            "exp1",
            "run1",
            ["score_collection_key_1"],
            ["score_collection_1"],
            {"score_collection_key_1": ["score_collection_1"]},
        ),
        (
            "exp1",
            "run1",
            ["score_collection_key_1", "score_collection_key_2"],
            ["score_collection_1", "score_collection_2"],
            {
                "score_collection_key_1": ["score_collection_1"],
                "score_collection_key_2": ["score_collection_2"],
            },
        ),
        (
            "exp1",
            "run1",
            ["score_collection_key_1", "score_collection_key_1"],
            ["score_collection_1", "score_collection_2"],
            {"score_collection_key_1": ["score_collection_1", "score_collection_2"]},
        ),
        (
            "exp1",
            "run1",
            ["score_collection_key_1", "score_collection_key_2", "score_collection_key_1"],
            ["score_collection_1", "score_collection_2", "score_collection_3"],
            {
                "score_collection_key_1": ["score_collection_1", "score_collection_3"],
                "score_collection_key_2": ["score_collection_2"],
            },
        ),
    ],
    indirect=["ls_score_collections", "expected_logs"],
)
def test_log_score_collection(
    in_memory_df_store: Dict[str, pd.DataFrame],
    score_collection_logger_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[FilesystemRunAdapter, FilesystemScoreCollectionLogger],
    ],
    experiment_id: str,
    run_id: str,
    ls_score_collection_names: List[str],
    ls_score_collections: List[Dict[str, float]],
    expected_logs: Dict[str, List[Dict[str, float]]],
):
    _, logger = score_collection_logger_factory(experiment_id, run_id)
    for name, score_dict in zip(ls_score_collection_names, ls_score_collections):
        logger.log(artifact_name=name, artifact=score_dict)
    for name, expected_rows in expected_logs.items():
        path = os.path.join(
            "mock_home_dir",
            "artifact_ml",
            experiment_id,
            run_id,
            "artifacts",
            "score_collections",
            name,
        )
        df = in_memory_df_store[path]
        expected_df = pd.DataFrame(expected_rows)
        pd.testing.assert_frame_equal(df.reset_index(drop=True), expected_df.reset_index(drop=True))
