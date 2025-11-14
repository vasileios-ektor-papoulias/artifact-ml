import os
from typing import Callable, Dict, List, Optional, Tuple

import pytest
from artifact_experiment._impl.neptune.adapter import NeptuneRunAdapter
from artifact_experiment._impl.neptune.loggers.plot_collections import (
    NeptunePlotCollectionLogger,
)
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_plot_collection_names, ls_plot_collections",
    [
        ("exp1", "run1", [], []),
        ("exp1", "run1", ["plot_collection_1"], ["plot_collection_1"]),
        (
            "exp1",
            "run1",
            ["plot_collection_1", "plot_collection_2"],
            ["plot_collection_1", "plot_collection_2"],
        ),
        (
            "exp1",
            "run1",
            ["plot_collection_1", "plot_collection_3"],
            ["plot_collection_1", "plot_collection_3"],
        ),
        (
            "exp1",
            "run1",
            ["plot_collection_1", "plot_collection_2", "plot_collection_3"],
            ["plot_collection_1", "plot_collection_2", "plot_collection_3"],
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
        ),
    ],
    indirect=["ls_plot_collections"],
)
def test_log(
    mocker: MockerFixture,
    plot_collection_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[NeptuneRunAdapter, NeptunePlotCollectionLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_plot_collection_names: List[str],
    ls_plot_collections: List[Dict[str, Figure]],
):
    adapter, logger = plot_collection_logger_factory(experiment_id, run_id)
    spy_adapter_log = mocker.spy(adapter, "log")
    for plot_collection_name, plot_collection in zip(ls_plot_collection_names, ls_plot_collections):
        logger.log(artifact_name=plot_collection_name, artifact=plot_collection)
    assert spy_adapter_log.call_count == len(ls_plot_collections)
    for idx, call_args in enumerate(spy_adapter_log.call_args_list):
        plot_collection_name = ls_plot_collection_names[idx]
        plot_collection = ls_plot_collections[idx]
        expected_path = os.path.join("artifacts", "plot_collections", plot_collection_name)
        assert call_args.kwargs == {"artifact_path": expected_path, "artifact": plot_collection}
