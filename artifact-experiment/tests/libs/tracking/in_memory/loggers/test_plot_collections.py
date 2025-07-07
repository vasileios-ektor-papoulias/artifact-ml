from typing import Callable, Dict, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryTrackingAdapter,
)
from artifact_experiment.libs.tracking.in_memory.loggers.plot_collections import (  # noqa: E501
    InMemoryPlotCollectionLogger,
)
from matplotlib.figure import Figure


@pytest.mark.parametrize(
    "ls_plot_collections",
    [
        ([]),
        (["plot_collection_1"]),
        (["plot_collection_1", "plot_collection_2"]),
        (["plot_collection_1", "plot_collection_3"]),
        (["plot_collection_1", "plot_collection_2", "plot_collection_3"]),
        (
            [
                "plot_collection_1",
                "plot_collection_2",
                "plot_collection_3",
                "plot_collection_4",
                "plot_collection_5",
            ]
        ),
    ],
    indirect=True,
)
def test_log(
    ls_plot_collections: List[Dict[str, Figure]],
    plot_collection_logger_factory: Callable[
        [Optional[InMemoryTrackingAdapter]],
        Tuple[InMemoryTrackingAdapter, InMemoryPlotCollectionLogger],
    ],
):
    adapter, logger = plot_collection_logger_factory(None)
    for idx, plot_collection in enumerate(ls_plot_collections, start=1):
        logger.log(artifact_name=f"plot_collection_{idx}", artifact=plot_collection)
    assert adapter.n_plot_collections == len(ls_plot_collections)
    for idx, expected_collection in enumerate(ls_plot_collections, start=1):
        key = f"test_experiment/test_run/plot_collections/plot_collection_{idx}/{idx}"
        stored_collection = adapter._native_run.dict_plot_collections[key]
        assert stored_collection.keys() == expected_collection.keys()
        for plot_name, expected_plot in expected_collection.items():
            assert stored_collection[plot_name] is expected_plot
