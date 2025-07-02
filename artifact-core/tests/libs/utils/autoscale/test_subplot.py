"""Tests for SubplotAutoscaler."""

import pytest
from artifact_core.libs.utils.autoscale.combined import (
    CombinedPlotAutoscaler,
    CombinedPlotScale,
    SubplotScaleConfig,
)


class TestSubplotScaleConfig:
    """Test cases for SubplotScaleConfig."""

    def test_default_values(self):
        """Test SubplotScaleConfig default values."""
        config = SubplotScaleConfig()

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
        assert config.features_per_base_subplot > 0

    def test_custom_values(self):
        """Test SubplotScaleConfig with custom values."""
        config = SubplotScaleConfig(
            base_font_size=18.0,
            base_figure_width=10.0,
            base_figure_height=8.0,
            features_per_base_subplot=12,
        )

        assert config.base_font_size == 18.0
        assert config.base_figure_width == 10.0
        assert config.base_figure_height == 8.0
        assert config.features_per_base_subplot == 12

    def test_is_frozen(self):
        """Test that SubplotScaleConfig is frozen (immutable)."""
        config = SubplotScaleConfig()

        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            config.base_font_size = 20.0


class TestSubplotScaleResult:
    """Test cases for SubplotScaleResult."""

    def test_creation(self):
        """Test SubplotScaleResult creation."""
        result = CombinedPlotScale(
            figure_width_scale=1.5,
            figure_height_scale=1.2,
            font_size=14.0,
            title_font_size=16.0,
            tick_font_size=12.0,
            legend_font_size=12.0,
            marker_size=5.0,
            line_width=2.0,
        )

        assert result.figure_width_scale == 1.5
        assert result.figure_height_scale == 1.2
        assert result.font_size == 14.0
        assert result.title_font_size == 16.0
        assert result.tick_font_size == 12.0
        assert result.legend_font_size == 12.0
        assert result.marker_size == 5.0
        assert result.line_width == 2.0

    def test_has_all_required_fields(self):
        """Test that result class has all required fields."""
        result = CombinedPlotScale(
            figure_width_scale=1.5,
            figure_height_scale=1.2,
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
        """Test that SubplotScaleResult is frozen (immutable)."""
        result = CombinedPlotScale(
            figure_width_scale=1.5,
            figure_height_scale=1.2,
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


class TestSubplotAutoscaler:
    """Test cases for SubplotAutoscaler."""

    @pytest.mark.parametrize(
        "num_subplots, expected_scale_increase",
        [
            (1, False),  # Single subplot - minimal scaling
            (4, False),  # Few subplots - minimal scaling
            (15, True),  # Many subplots - some scaling
            (30, True),  # Very many subplots - significant scaling
            (100, True),  # Extreme number - max scaling
        ],
    )
    def test_calculate_scale(self, num_subplots: int, expected_scale_increase: bool):
        """Test subplot scaling behavior with various subplot counts."""
        config = SubplotScaleConfig()
        result = CombinedPlotAutoscaler.compute(num_subplots, config)

        # Check return type
        assert isinstance(result, CombinedPlotScale)

        # Check scale bounds
        assert result.figure_width_scale >= 1.0
        assert result.figure_height_scale >= 1.0
        assert result.font_size > 0

        if expected_scale_increase:
            assert result.figure_width_scale > 1.0
            assert result.figure_height_scale > 1.0

    def test_calculate_scale_with_custom_config(self):
        """Test SubplotAutoscaler with custom configuration."""
        custom_config = SubplotScaleConfig(
            base_font_size=18.0,
            base_figure_width=10.0,
            base_figure_height=8.0,
            features_per_base_subplot=12,
        )

        result = CombinedPlotAutoscaler.compute(24, custom_config)

        # Should use custom configuration
        assert result.figure_width_scale <= custom_config.max_scale_factor
        assert result.figure_height_scale <= custom_config.max_scale_factor

    def test_calculate_scale_with_none_config(self):
        """Test SubplotAutoscaler creates default config when None."""
        result = CombinedPlotAutoscaler.compute(15, None)

        assert isinstance(result, CombinedPlotScale)
        assert result.figure_width_scale >= 1.0
        assert result.figure_height_scale >= 1.0

    def test_calculate_scale_conservative_scaling(self):
        """Test SubplotAutoscaler applies conservative scaling."""
        config = SubplotScaleConfig()

        # Test that scaling is more conservative than pure proportional
        result_moderate = CombinedPlotAutoscaler.compute(18, config)
        result_many = CombinedPlotAutoscaler.compute(36, config)

        # Conservative scaling means less than 2x increase for 2x subplots
        scale_ratio = result_many.figure_width_scale / result_moderate.figure_width_scale
        assert scale_ratio < 2.0

    def test_calculate_scale_edge_cases(self):
        """Test SubplotAutoscaler with edge cases."""
        config = SubplotScaleConfig()

        # Zero subplots
        result = CombinedPlotAutoscaler.compute(0, config)
        assert result.figure_width_scale >= 1.0
        assert result.figure_height_scale >= 1.0

        # Very large number of subplots
        result = CombinedPlotAutoscaler.compute(1000, config)
        assert result.figure_width_scale <= config.max_scale_factor
        assert result.figure_height_scale <= config.max_scale_factor

    def test_calculate_scale_consistency(self):
        """Test that scaling is consistent and predictable."""
        config = SubplotScaleConfig()

        # More subplots should have larger or equal scaling
        result_small = CombinedPlotAutoscaler.compute(5, config)
        result_large = CombinedPlotAutoscaler.compute(20, config)

        assert result_small.figure_width_scale <= result_large.figure_width_scale
        assert result_small.figure_height_scale <= result_large.figure_height_scale

    def test_calculate_scale_bounds_respected(self):
        """Test that scale factors respect min/max bounds."""
        config = SubplotScaleConfig(min_scale_factor=0.4, max_scale_factor=2.0)

        # Test with extreme values
        result_small = CombinedPlotAutoscaler.compute(1, config)
        result_large = CombinedPlotAutoscaler.compute(1000, config)

        # Scaling should respect max bounds
        assert result_large.figure_width_scale <= config.max_scale_factor
        assert result_large.figure_height_scale <= config.max_scale_factor

        # Font scaling should respect min bounds
        assert result_large.font_size >= config.base_font_size * config.min_scale_factor

    def test_calculate_scale_symmetric_scaling(self):
        """Test that width and height scale symmetrically."""
        config = SubplotScaleConfig()

        # For most cases, width and height should scale similarly
        result = CombinedPlotAutoscaler.compute(25, config)

        # Width and height scaling should be close (within 20% difference)
        ratio = result.figure_width_scale / result.figure_height_scale
        assert 0.8 <= ratio <= 1.2
