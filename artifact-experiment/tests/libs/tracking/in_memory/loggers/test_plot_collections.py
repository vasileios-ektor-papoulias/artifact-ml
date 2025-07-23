from typing import Callable, Dict, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryRunAdapter,
)
from artifact_experiment.libs.tracking.in_memory.loggers.plot_collections import (
    InMemoryPlotCollectionLogger,
)
from matplotlib.figure import Figure


@pytest.mark.parametrize(
    "experiment_id, run_id, ls_plot_collections",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["plot_collection_1"]),
        ("exp1", "run1", ["plot_collection_1", "plot_collection_2"]),
        ("exp1", "run1", ["plot_collection_1", "plot_collection_3"]),
        ("exp1", "run1", ["plot_collection_1", "plot_collection_2", "plot_collection_3"]),
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
        ),
    ],
    indirect=["ls_plot_collections"],
)
def test_log(
    plot_collection_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryPlotCollectionLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_plot_collections: List[Dict[str, Figure]],
):
    adapter, logger = plot_collection_logger_factory(experiment_id, run_id)
    for idx, plot_collection in enumerate(ls_plot_collections, start=1):
        logger.log(artifact_name=f"plot_collection_{idx}", artifact=plot_collection)
    assert adapter.n_plot_collections == len(ls_plot_collections)
    for idx, expected_collection in enumerate(ls_plot_collections, start=1):
        key = f"{experiment_id}/{run_id}/plot_collections/plot_collection_{idx}/1"
        stored_collection = adapter.dict_plot_collections[key]
        assert stored_collection.keys() == expected_collection.keys()
        for plot_name, _ in expected_collection.items():
            assert isinstance(stored_collection[plot_name], Figure)
