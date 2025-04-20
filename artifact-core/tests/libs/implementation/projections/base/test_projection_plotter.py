from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
from artifact_core.libs.implementation.projections.base.plotter import (
    ProjectionPlotter,
    ProjectionPlotterConfig,
)
from artifact_core.libs.utils.plot_combiner import PlotCombinationConfig
from matplotlib.figure import Figure


@pytest.fixture
def set_agg_backend():
    import matplotlib

    matplotlib.use("Agg")


@pytest.fixture
def close_all_figs_after_test():
    yield
    plt.close("all")


@pytest.fixture
def projection_2d_real() -> np.ndarray:
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])


@pytest.fixture
def projection_2d_synthetic() -> np.ndarray:
    return np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5], [7.5, 8.5], [9.5, 10.5]])


@pytest.fixture
def plotter_factory() -> Callable[[str], ProjectionPlotter]:
    def _factory(plotter_type: str = "default") -> ProjectionPlotter:
        if plotter_type == "custom":
            config = ProjectionPlotterConfig(
                scatter_color="blue",
                failed_suffix="Custom failed message",
                figsize=(8, 8),
                title_prefix="Custom Projection",
                combined_config=PlotCombinationConfig(
                    n_cols=1,
                    dpi=100,
                    combined_title="Custom Combined Title",
                ),
            )
            return ProjectionPlotter(config=config)
        else:
            return ProjectionPlotter(config=None)

    return _factory


@pytest.mark.parametrize(
    "use_projection_data, plotter_type, projection_name, "
    + "expected_title_contains, expected_axis_labels, expect_has_collections, expected_figsize",
    [
        (
            True,
            "default",
            "PCA",
            "2D Projection: PCA",
            {"x": "Dim 1", "y": "Dim 2"},
            True,
            None,
        ),
        (
            False,
            "default",
            "PCA",
            None,
            None,
            False,
            None,
        ),
        (
            True,
            "custom",
            "PCA",
            "Custom Projection: PCA",
            {"x": "Dim 1", "y": "Dim 2"},
            True,
            (8, 8),
        ),
    ],
)
def test_produce_projection_plot(
    set_agg_backend,
    close_all_figs_after_test,
    projection_2d_real: np.ndarray,
    plotter_factory: Callable[[str], ProjectionPlotter],
    use_projection_data: bool,
    plotter_type: str,
    projection_name: str,
    expected_title_contains: str,
    expected_axis_labels: Dict[str, str],
    expect_has_collections: bool,
    expected_figsize: Optional[Tuple[float, float]],
):
    projection_data = None if not use_projection_data else projection_2d_real
    plotter = plotter_factory(plotter_type)
    result = plotter.produce_projection_plot(
        dataset_projection_2d=projection_data,
        projection_name=projection_name,
    )
    assert isinstance(result, Figure), "Result should be a Figure"
    assert result.get_axes(), "Figure should have at least one axis"
    ax = result.axes[0]
    if projection_data is None:
        ax_title = ax.get_title()
        expected_title = f"{projection_name}: Projection failed (rank or numeric issues)."
        assert ax_title == expected_title, f"Expected title '{expected_title}', got '{ax_title}'"
        assert not ax.axison, "Axis should be turned off for None projection"
    else:
        assert len(result.texts) >= 1, "Figure should have at least one text element (title)"
        title_text = result.texts[0].get_text() if result.texts else ""
        assert expected_title_contains in title_text, (
            f"Expected title to contain '{expected_title_contains}', got '{title_text}'"
        )
        if expected_axis_labels:
            assert ax.get_xlabel() == expected_axis_labels["x"], (
                f"Expected x-label '{expected_axis_labels['x']}', got '{ax.get_xlabel()}'"
            )
            assert ax.get_ylabel() == expected_axis_labels["y"], (
                f"Expected y-label '{expected_axis_labels['y']}', got '{ax.get_ylabel()}'"
            )
        if expect_has_collections:
            assert len(ax.collections) > 0, "Axis should have collections (scatter)"
        if expected_figsize:
            width, height = result.get_size_inches()
            assert (width, height) == expected_figsize, (
                f"Expected figure size {expected_figsize}, got ({width}, {height})"
            )


@pytest.mark.parametrize(
    "use_real_data, use_synthetic_data, plotter_type, projection_name, "
    + "expected_title, expected_axes_count, expected_dpi",
    [
        (
            True,
            True,
            "default",
            "PCA",
            "2D Projection Comparison: PCA",
            2,
            None,
        ),
        (
            True,
            False,
            "default",
            "PCA",
            "2D Projection Comparison: PCA",
            2,
            None,
        ),
        (
            True,
            True,
            "custom",
            "PCA",
            "Custom Combined Title: PCA",
            2,
            100,
        ),
    ],
)
def test_produce_projection_comparison_plot(
    set_agg_backend,
    close_all_figs_after_test,
    projection_2d_real: np.ndarray,
    projection_2d_synthetic: np.ndarray,
    plotter_factory: Callable[[str], ProjectionPlotter],
    use_real_data: bool,
    use_synthetic_data: bool,
    plotter_type: str,
    projection_name: str,
    expected_title: str,
    expected_axes_count: int,
    expected_dpi: Optional[float],
):
    real_projection_data = None if use_real_data else projection_2d_real
    synthetic_projection_data = None if use_synthetic_data else projection_2d_synthetic
    plotter = plotter_factory(plotter_type)
    result = plotter.produce_projection_comparison_plot(
        dataset_projection_2d_real=real_projection_data,
        dataset_projection_2d_synthetic=synthetic_projection_data,
        projection_name=projection_name,
    )
    assert isinstance(result, Figure), "Result should be a Figure"
    assert result.get_axes(), "Figure should have at least one axis"
    assert len(result.texts) >= 1, "Figure should have at least one text element (title)"
    title_text = result.texts[0].get_text() if result.texts else ""
    assert expected_title in title_text, (
        f"Expected title to contain '{expected_title}', got '{title_text}'"
    )
    assert len(result.axes) == expected_axes_count, (
        f"Expected {expected_axes_count} axes, got {len(result.axes)}"
    )
    for i, ax in enumerate(result.axes):
        assert ax is not None, f"Subplot {i} should exist"
    if expected_dpi:
        assert result.dpi == expected_dpi, f"Expected DPI {expected_dpi}, got {result.dpi}"
