"""Tests for CategoricalAutoscaler."""

import pytest
from artifact_core.libs.utils.autoscale.categorical import (
    CategoricalAutoscaler,
    CategoricalScaleConfig,
    CategoricalScaleResult,
)


class TestCategoricalScaleConfig:
    """Test cases for CategoricalScaleConfig."""

    def test_default_values(self):
        """Test CategoricalScaleConfig default values."""
        config = CategoricalScaleConfig()

        # Check all attributes exist with reasonable defaults
        assert config.base_figure_width > 0
        assert config.base_figure_height > 0
        assert config.base_font_size > 0
        assert config.base_title_font_size > 0
        assert config.base_tick_font_size > 0
        assert config.base_legend_font_size > 0
        assert config.base_marker_size > 0
        assert config.base_line_width > 0
        assert 0 < config.min_scale_factor <= 1.0
        assert config.max_scale_factor >= 1.0
        assert config.categories_per_base_width > 0

    def test_custom_values(self):
        """Test CategoricalScaleConfig with custom values."""
        config = CategoricalScaleConfig(
            base_font_size=20.0,
            base_figure_width=8.0,
            max_scale_factor=5.0,
            categories_per_base_width=3,
        )

        assert config.base_font_size == 20.0
        assert config.base_figure_width == 8.0
        assert config.max_scale_factor == 5.0
        assert config.categories_per_base_width == 3

    def test_is_frozen(self):
        """Test that CategoricalScaleConfig is frozen (immutable)."""
        config = CategoricalScaleConfig()

        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            config.base_font_size = 20.0


class TestCategoricalScaleResult:
    """Test cases for CategoricalScaleResult."""

    def test_creation(self):
        """Test CategoricalScaleResult creation."""
        result = CategoricalScaleResult(
            figure_width_scale=1.5,
            figure_height_scale=1.0,
            font_size=14.0,
            title_font_size=16.0,
            tick_font_size=12.0,
            legend_font_size=12.0,
            marker_size=5.0,
            line_width=2.0,
        )

        assert result.figure_width_scale == 1.5
        assert result.figure_height_scale == 1.0
        assert result.font_size == 14.0
        assert result.title_font_size == 16.0
        assert result.tick_font_size == 12.0
        assert result.legend_font_size == 12.0
        assert result.marker_size == 5.0
        assert result.line_width == 2.0

    def test_has_all_required_fields(self):
        """Test that result class has all required fields."""
        result = CategoricalScaleResult(
            figure_width_scale=1.5,
            figure_height_scale=1.0,
            font_size=14.0,
            title_font_size=16.0,
            tick_font_size=12.0,
            legend_font_size=12.0,
            marker_size=5.0,
            line_width=2.0,
        )

        required_fields = [
            "figure_width_scale",
            "figure_height_scale",
            "font_size",
            "title_font_size",
            "tick_font_size",
            "legend_font_size",
            "marker_size",
            "line_width",
        ]

        assert all(hasattr(result, field) for field in required_fields)

    def test_is_frozen(self):
        """Test that CategoricalScaleResult is frozen (immutable)."""
        result = CategoricalScaleResult(
            figure_width_scale=1.5,
            figure_height_scale=1.0,
            font_size=14.0,
            title_font_size=16.0,
            tick_font_size=12.0,
            legend_font_size=12.0,
            marker_size=5.0,
            line_width=2.0,
        )

        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            result.figure_width_scale = 2.0


class TestCategoricalAutoscaler:
    """Test cases for CategoricalAutoscaler."""

    @pytest.mark.parametrize(
        "num_categories, expected_width_increase, expected_font_decrease",
        [
            (1, False, False),  # Single category - minimal scaling
            (3, False, False),  # Few categories - minimal scaling
            (10, True, True),  # Moderate categories - some scaling
            (25, True, True),  # Many categories - significant scaling
            (50, True, True),  # Very many categories - max scaling
        ],
    )
    def test_calculate_scale(
        self, num_categories: int, expected_width_increase: bool, expected_font_decrease: bool
    ):
        """Test categorical scaling behavior with various category counts."""
        config = CategoricalScaleConfig()
        result = CategoricalAutoscaler.compute(num_categories, config)

        # Check return type
        assert isinstance(result, CategoricalScaleResult)

        # Check scale bounds
        assert result.figure_width_scale >= 1.0
        assert result.figure_height_scale == 1.0  # Height doesn't change
        assert result.font_size > 0
        assert result.title_font_size > 0
        assert result.tick_font_size > 0
        assert result.legend_font_size > 0
        assert result.marker_size > 0
        assert result.line_width > 0

        # Check scaling logic
        if expected_width_increase:
            assert result.figure_width_scale > 1.0

        if expected_font_decrease:
            assert result.font_size < config.base_font_size

    def test_calculate_scale_with_custom_config(self):
        """Test CategoricalAutoscaler with custom configuration."""
        custom_config = CategoricalScaleConfig(
            base_font_size=20.0,
            base_figure_width=8.0,
            max_scale_factor=5.0,
            categories_per_base_width=3,
        )

        result = CategoricalAutoscaler.compute(15, custom_config)

        # Should use custom base values
        assert result.figure_width_scale <= custom_config.max_scale_factor

        # Font scaling should be based on custom base
        expected_min_font = custom_config.base_font_size * 0.5
        assert result.font_size >= expected_min_font

    def test_calculate_scale_with_none_config(self):
        """Test CategoricalAutoscaler creates default config when None."""
        result = CategoricalAutoscaler.compute(10, None)

        assert isinstance(result, CategoricalScaleResult)
        assert result.figure_width_scale >= 1.0
        assert result.font_size > 0

    def test_calculate_scale_edge_cases(self):
        """Test CategoricalAutoscaler with edge cases."""
        config = CategoricalScaleConfig()

        # Zero categories
        result = CategoricalAutoscaler.compute(0, config)
        assert result.figure_width_scale >= 1.0

        # Very large number of categories
        result = CategoricalAutoscaler.compute(1000, config)
        assert result.figure_width_scale <= config.max_scale_factor
        assert result.font_size >= config.base_font_size * config.min_scale_factor

    def test_calculate_scale_consistency(self):
        """Test that scaling is consistent and predictable."""
        config = CategoricalScaleConfig()

        # Smaller inputs should have smaller or equal scaling
        result_small = CategoricalAutoscaler.compute(5, config)
        result_large = CategoricalAutoscaler.compute(20, config)

        assert result_small.figure_width_scale <= result_large.figure_width_scale
        assert result_small.font_size >= result_large.font_size

    def test_calculate_scale_height_unchanged(self):
        """Test that height scaling is always 1.0 for categorical plots."""
        config = CategoricalScaleConfig()
        test_cases = [1, 5, 10, 25, 50, 100]

        for num_categories in test_cases:
            result = CategoricalAutoscaler.compute(num_categories, config)
            assert result.figure_height_scale == 1.0

    def test_calculate_scale_bounds_respected(self):
        """Test that scale factors respect min/max bounds."""
        config = CategoricalScaleConfig(min_scale_factor=0.3, max_scale_factor=2.5)

        # Test with extreme values
        result_small = CategoricalAutoscaler.compute(1, config)
        result_large = CategoricalAutoscaler.compute(1000, config)

        # Width scaling should respect max bounds
        assert result_large.figure_width_scale <= config.max_scale_factor

        # Font scaling should respect min bounds
        assert result_large.font_size >= config.base_font_size * config.min_scale_factor
