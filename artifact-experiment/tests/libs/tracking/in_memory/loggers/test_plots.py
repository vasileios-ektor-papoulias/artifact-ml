from typing import Callable, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryTrackingAdapter,
)
from artifact_experiment.libs.tracking.in_memory.loggers.plots import (
    InMemoryPlotLogger,
)
from matplotlib.figure import Figure


@pytest.mark.parametrize(
    "ls_plots",
    [
        ([]),
        (["plot_1"]),
        (["plot_1", "plot_2"]),
        (["plot_1", "plot_3"]),
        (["plot_1", "plot_2", "plot_3"]),
        (["plot_1", "plot_2", "plot_3", "plot_4", "plot_5"]),
    ],
    indirect=True,
)
def test_log(
    ls_plots: List[Figure],
    plot_logger_factory: Callable[
        [Optional[InMemoryTrackingAdapter]], Tuple[InMemoryTrackingAdapter, InMemoryPlotLogger]
    ],
):
    adapter, logger = plot_logger_factory(None)
    for idx, plot in enumerate(ls_plots, start=1):
        logger.log(artifact_name=f"plot_{idx}", artifact=plot)
    assert adapter.n_plots == len(ls_plots)
    for idx, expected_plot in enumerate(ls_plots, start=1):
        key = f"test_experiment/test_run/plots/plot_{idx}/{idx}"
        stored_plot = adapter._native_run.dict_plots[key]
        assert stored_plot is expected_plot
