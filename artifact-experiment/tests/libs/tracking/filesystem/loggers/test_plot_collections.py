import os
from typing import Callable, Dict, List, Optional, Tuple
from unittest.mock import ANY

import pytest
from artifact_experiment.libs.tracking.filesystem.adapter import FilesystemRunAdapter
from artifact_experiment.libs.tracking.filesystem.loggers.plot_collections import (
    FilesystemPlotCollectionLogger,
)
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_plot_collection_names, ls_plot_collections, ls_step",
    [
        ("exp1", "run1", [], [], []),
        ("exp1", "run1", ["plot_collection_1"], ["plot_collection_1"], [1]),
        (
            "exp1",
            "run1",
            ["plot_collection_1", "plot_collection_2"],
            ["plot_collection_1", "plot_collection_2"],
            [1, 1],
        ),
        (
            "exp1",
            "run1",
            ["plot_collection_1", "plot_collection_3"],
            ["plot_collection_1", "plot_collection_3"],
            [1, 1],
        ),
        (
            "exp1",
            "run1",
            ["plot_collection_1", "plot_collection_2", "plot_collection_3"],
            ["plot_collection_1", "plot_collection_2", "plot_collection_3"],
            [1, 1, 1],
        ),
        (
            "exp1",
            "run1",
            [
                "plot_collection_1",
                "plot_collection_2",
                "plot_collection_3",
                "plot_collection_4",
                "plot_collection_5",
            ],
            [
                "plot_collection_1",
                "plot_collection_2",
                "plot_collection_3",
                "plot_collection_4",
                "plot_collection_5",
            ],
            [1, 1, 1, 1, 1],
        ),
        (
            "exp1",
            "run1",
            ["plot_collection_1", "plot_collection_2", "plot_collection_1"],
            ["plot_collection_1", "plot_collection_2", "plot_collection_3"],
            [1, 1, 2],
        ),
        (
            "exp1",
            "run1",
            [
                "plot_collection_1",
                "plot_collection_1",
                "plot_collection_2",
                "plot_collection_2",
                "plot_collection_2",
                "plot_collection_1",
                "plot_collection_3",
            ],
            [
                "plot_collection_1",
                "plot_collection_2",
                "plot_collection_3",
                "plot_collection_1",
                "plot_collection_2",
                "plot_collection_3",
                "plot_collection_5",
            ],
            [1, 2, 1, 2, 3, 3, 1],
        ),
    ],
    indirect=["ls_plot_collections"],
)
def test_log_plot_collection(
    mocker: MockerFixture,
    mock_incremental_path_generator: List[str],
    plot_collection_logger_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[FilesystemRunAdapter, FilesystemPlotCollectionLogger],
    ],
    experiment_id: str,
    run_id: str,
    ls_plot_collection_names: List[str],
    ls_plot_collections: List[Dict[str, Figure]],
    ls_step: List[int],
):
    _, logger = plot_collection_logger_factory(experiment_id, run_id)
    savefig_mock = mocker.patch.object(Figure, "savefig")
    for name, plots in zip(ls_plot_collection_names, ls_plot_collections):
        logger.log(artifact_name=name, artifact=plots)
    assert len(mock_incremental_path_generator) == len(ls_plot_collections)
    for i, (name, plot_collection, step) in enumerate(
        zip(ls_plot_collection_names, ls_plot_collections, ls_step)
    ):
        expected_dir = os.path.join(
            "mock_home_dir",
            "artifact_ml",
            experiment_id,
            run_id,
            "artifacts",
            "plot_collections",
            name,
            str(step),
        )
        assert mock_incremental_path_generator[i] == expected_dir
        for plot_name in plot_collection.keys():
            filename = f"{plot_name}.png"
            expected_path = os.path.join(expected_dir, filename)
            savefig_mock.assert_any_call(fname=expected_path, dpi=ANY, bbox_inches=ANY, format=ANY)
