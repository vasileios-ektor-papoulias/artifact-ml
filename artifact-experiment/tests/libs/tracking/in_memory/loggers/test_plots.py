from typing import Callable, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryRunAdapter,
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
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryPlotLogger]
    ],
):
    adapter, logger = plot_logger_factory("test_experiment", "test_run")
    for idx, plot in enumerate(ls_plots, start=1):
        logger.log(artifact_name=f"plot_{idx}", artifact=plot)
    assert adapter.n_plots == len(ls_plots)
    for idx, _ in enumerate(ls_plots, start=1):
        key = f"test_experiment/test_run/plots/plot_{idx}/1"
        stored_plot = adapter.dict_plots[key]
        assert isinstance(stored_plot, Figure)
