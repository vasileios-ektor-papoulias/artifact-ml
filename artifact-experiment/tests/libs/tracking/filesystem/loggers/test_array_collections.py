import os
from typing import Callable, Dict, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.filesystem.adapter import FilesystemRunAdapter
from artifact_experiment.libs.tracking.filesystem.loggers.array_collections import (
    FilesystemArrayCollectionLogger,
)
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_array_collection_names, ls_array_collections, ls_step",
    [
        ("exp1", "run1", [], [], []),
        ("exp1", "run1", ["array_collection_1"], ["array_collection_1"], [1]),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_2"],
            ["array_collection_1", "array_collection_2"],
            [1, 1],
        ),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_3"],
            ["array_collection_1", "array_collection_3"],
            [1, 1],
        ),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_2", "array_collection_3"],
            ["array_collection_1", "array_collection_2", "array_collection_3"],
            [1, 1, 1],
        ),
        (
            "exp1",
            "run1",
            [
                "array_collection_1",
                "array_collection_2",
                "array_collection_3",
                "array_collection_4",
                "array_collection_5",
            ],
            [
                "array_collection_1",
                "array_collection_2",
                "array_collection_3",
                "array_collection_4",
                "array_collection_5",
            ],
            [1, 1, 1, 1, 1],
        ),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_2", "array_collection_1"],
            ["array_collection_1", "array_collection_2", "array_collection_3"],
            [1, 1, 2],
        ),
        (
            "exp1",
            "run1",
            [
                "array_collection_1",
                "array_collection_1",
                "array_collection_2",
                "array_collection_2",
                "array_collection_2",
                "array_collection_1",
                "array_collection_3",
            ],
            [
                "array_collection_1",
                "array_collection_2",
                "array_collection_3",
                "array_collection_1",
                "array_collection_2",
                "array_collection_3",
                "array_collection_5",
            ],
            [1, 2, 1, 2, 3, 3, 1],
        ),
    ],
    indirect=["ls_array_collections"],
)
def test_log_array_collection(
    mocker: MockerFixture,
    mock_incremental_path_generator: List[str],
    array_collection_logger_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[FilesystemRunAdapter, FilesystemArrayCollectionLogger],
    ],
    experiment_id: str,
    run_id: str,
    ls_array_collection_names: List[str],
    ls_array_collections: List[Dict[str, Array]],
    ls_step: List[int],
):
    _, logger = array_collection_logger_factory(experiment_id, run_id)
    mock_savez = mocker.patch("numpy.savez_compressed")
    for name, coll in zip(ls_array_collection_names, ls_array_collections):
        logger.log(artifact_name=name, artifact=coll)
    assert mock_savez.call_count == len(ls_array_collections)
    assert len(mock_incremental_path_generator) == len(ls_array_collections)
    for i, (name, array_collection, step) in enumerate(
        zip(ls_array_collection_names, ls_array_collections, ls_step)
    ):
        expected_dir = os.path.join(
            "mock_home_dir",
            "artifact_ml",
            experiment_id,
            run_id,
            "artifacts",
            "array_collections",
            name,
        )
        expected_path = os.path.join(expected_dir, f"{step}.npz")
        assert mock_incremental_path_generator[i] == expected_path
        mock_savez.assert_any_call(file=expected_path, allow_pickle=True, **array_collection)
