from typing import Dict, Optional, Tuple

import numpy as np
import pytest
from artifact_core.libs.implementation.tabular.projections.base.plotter import (
    ProjectionPlotter,
)
from matplotlib.figure import Figure


@pytest.mark.unit
@pytest.mark.parametrize(
    "projection_2d_real_dispatcher, projection_plotter_dispatcher, projection_name, "
    + "expected_title_contains, expected_axis_labels, expect_has_collections, expected_figsize",
    [
        (
            "projection_2d_real",
            "default_projection_plotter",
            "PCA",
            "2D Projection: PCA",
            {"x": "Dim 1", "y": "Dim 2"},
            True,
            None,
        ),
        (
            "null",
            "default_projection_plotter",
            "PCA",
            "",
            {},
            False,
            None,
        ),
        (
            "projection_2d_real",
            "custom_projection_plotter",
            "PCA",
            "Custom Projection: PCA",
            {"x": "Dim 1", "y": "Dim 2"},
            True,
            (8, 8),
        ),
    ],
    indirect=["projection_2d_real_dispatcher", "projection_plotter_dispatcher"],
)
def test_produce_projection_plot(
    set_agg_backend,
    close_all_figs_after_test,
    projection_2d_real_dispatcher: Optional[np.ndarray],
    projection_plotter_dispatcher: ProjectionPlotter,
    projection_name: str,
    expected_title_contains: str,
    expected_axis_labels: Dict[str, str],
    expect_has_collections: bool,
    expected_figsize: Optional[Tuple[float, float]],
):
    projection_data = projection_2d_real_dispatcher
    plotter = projection_plotter_dispatcher
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
        if expected_title_contains:
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "projection_2d_real_dispatcher, projection_2d_synthetic_dispatcher, "
    + "projection_plotter_dispatcher, projection_name, "
    + "expected_title, expected_axes_count, expected_dpi",
    [
        (
            "projection_2d_real",
            "projection_2d_synthetic",
            "default_projection_plotter",
            "PCA",
            "2D Projection Comparison: PCA",
            2,
            None,
        ),
        (
            "projection_2d_real",
            "null",
            "default_projection_plotter",
            "PCA",
            "2D Projection Comparison: PCA",
            2,
            None,
        ),
        (
            "projection_2d_real",
            "projection_2d_synthetic",
            "custom_projection_plotter",
            "PCA",
            "Custom Combined Title: PCA",
            2,
            100,
        ),
    ],
    indirect=[
        "projection_2d_real_dispatcher",
        "projection_2d_synthetic_dispatcher",
        "projection_plotter_dispatcher",
    ],
)
def test_produce_projection_comparison_plot(
    set_agg_backend,
    close_all_figs_after_test,
    projection_2d_real_dispatcher: Optional[np.ndarray],
    projection_2d_synthetic_dispatcher: Optional[np.ndarray],
    projection_plotter_dispatcher: ProjectionPlotter,
    projection_name: str,
    expected_title: str,
    expected_axes_count: int,
    expected_dpi: Optional[float],
):
    real_projection_data = projection_2d_real_dispatcher
    synthetic_projection_data = projection_2d_synthetic_dispatcher
    plotter = projection_plotter_dispatcher
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
