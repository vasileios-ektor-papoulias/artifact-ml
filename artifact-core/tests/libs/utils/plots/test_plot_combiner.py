from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt
import pytest
from artifact_core._libs.artifacts.tools.plotters.plot_combiner import (
    PlotCombinationConfig,
    PlotCombiner,
)


@pytest.fixture
def set_agg_backend():
    import matplotlib

    matplotlib.use("Agg")


@pytest.fixture
def close_all_figs_after_test():
    yield
    plt.close("all")


@pytest.fixture
def sample_figures_factory():
    def _factory(num_figs: int) -> Dict[str, Figure]:
        figs = {}
        for i in range(num_figs):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [i, i + 1, i + 2], label=f"Line {i}")
            ax.set_title(f"Sample Figure {i}")
            figs[f"figure_{i}"] = fig
        return figs

    return _factory


@pytest.mark.unit
@pytest.mark.parametrize(
    "num_figs, n_cols, combined_title, expected_num_axes",
    [
        (0, 2, None, 0),
        (1, 1, None, 1),
        (3, 2, None, 4),
        (3, 2, "My Combined Title", 4),
        (1, 2, None, 2),
        (2, 2, None, 2),
        (5, 2, "Large Title", 6),
        (6, 3, None, 6),
    ],
)
def test_plot_combiner_factory(
    set_agg_backend,
    close_all_figs_after_test,
    sample_figures_factory: Callable[..., Dict[str, Figure]],
    num_figs: int,
    n_cols: int,
    combined_title: Optional[str],
    expected_num_axes: int,
):
    dict_plots = sample_figures_factory(num_figs)
    config = PlotCombinationConfig(n_cols=n_cols, combined_title=combined_title)
    combined_fig = PlotCombiner.combine(dict_plots=dict_plots, config=config)
    assert len(combined_fig.axes) == expected_num_axes, (
        f"Expected {expected_num_axes} Axes for {num_figs} figs with {n_cols} cols"
    )
    if len(dict_plots) < expected_num_axes:
        for ax_idx, ax in enumerate(combined_fig.axes[len(dict_plots) :], start=len(dict_plots)):
            assert not ax.get_images(), f"Extra Ax (index {ax_idx}) should not contain any image"
    if combined_title is not None:
        assert any(t.get_text() == combined_title for t in combined_fig.texts), (
            f"Expected combined title '{combined_title}' not found in figure texts."
        )
