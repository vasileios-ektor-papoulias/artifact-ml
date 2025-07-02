"""Tests for GridAutoscaler."""

import pytest

from artifact_core.libs.utils.autoscale.grid import GridAutoscaler, GridAutoscalerHyperparams, GridScaleResult


class TestGridScaleConfig:
    """Test cases for GridScaleConfig."""

    def test_default_values(self):
        """Test GridScaleConfig default values."""
        config = GridAutoscalerHyperparams()

        # Check all attributes exist with reasonable defaults
        assert config.base_figure_width > 0
        assert config.base_figure_height > 0
        assert config.base_font_size > 0
        assert config.base_title_font_size > 0
        assert config.base_tick_font_size > 0
        assert config.base_legend_font_size > 0
        assert config.base_annotation_font_size > 0
        assert config.base_marker_size > 0
        assert config.base_line_width > 0
        assert 0 < config.min_scale_factor <= 1.0
        assert config.max_scale_factor >= 1.0
        assert config.grid_cells_per_base_size > 0

    def test_custom_values(self):
        """Test GridScaleConfig with custom values."""
        config = GridAutoscalerHyperparams(
            base_figure_width=9.0,
            base_figure_height=9.0,
            base_annotation_font_size=10.0,
            grid_cells_per_base_size=5,
        )

        assert config.base_figure_width == 9.0
        assert config.base_figure_height == 9.0
        assert config.base_annotation_font_size == 10.0
        assert config.grid_cells_per_base_size == 5

    def test_is_frozen(self):
        """Test that GridScaleConfig is frozen (immutable)."""
        config = GridAutoscalerHyperparams()

        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            config.base_font_size = 20.0


class TestGridScaleResult:
    """Test cases for GridScaleResult."""

    def test_creation(self):
        """Test GridScaleResult creation."""
        result = GridScaleResult(
            figure_width_scale=1.5,
            figure_height_scale=1.5,
            font_size=14.0,
            title_font_size=16.0,
            tick_font_size=12.0,
            legend_font_size=12.0,
            annotation_font_size=8.0,
            marker_size=5.0,
            line_width=2.0,
        )

        assert result.figure_width_scale == 1.5
        assert result.figure_height_scale == 1.5
        assert result.font_size == 14.0
        assert result.title_font_size == 16.0
        assert result.tick_font_size == 12.0
        assert result.legend_font_size == 12.0
        assert result.annotation_font_size == 8.0
        assert result.marker_size == 5.0
        assert result.line_width == 2.0

    def test_has_all_required_fields(self):
        """Test that result class has all required fields."""
        result = GridScaleResult(
            figure_width_scale=1.5,
            figure_height_scale=1.5,
            font_size=14.0,
            title_font_size=16.0,
            tick_font_size=12.0,
            legend_font_size=12.0,
            annotation_font_size=8.0,
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
            "annotation_font_size",
            "marker_size",
            "line_width",
        ]

        assert all(hasattr(result, field) for field in required_fields)

    def test_has_annotation_font_size(self):
        """Test that GridScaleResult has annotation_font_size field."""
        result = GridScaleResult(
            figure_width_scale=1.5,
            figure_height_scale=1.5,
            font_size=14.0,
            title_font_size=16.0,
            tick_font_size=12.0,
            legend_font_size=12.0,
            annotation_font_size=8.0,
            marker_size=5.0,
            line_width=2.0,
        )

        assert hasattr(result, "annotation_font_size")
        assert result.annotation_font_size == 8.0

    def test_is_frozen(self):
        """Test that GridScaleResult is frozen (immutable)."""
        result = GridScaleResult(
            figure_width_scale=1.5,
            figure_height_scale=1.5,
            font_size=14.0,
            title_font_size=16.0,
            tick_font_size=12.0,
            legend_font_size=12.0,
            annotation_font_size=8.0,
            marker_size=5.0,
            line_width=2.0,
        )

        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            result.figure_width_scale = 2.0


class TestGridAutoscaler:
    """Test cases for GridAutoscaler."""

    @pytest.mark.parametrize(
        "grid_size, expected_scale_increase",
        [
            ((5, 5), False),  # Small grid (5x5) - minimal scaling
            ((10, 10), True),  # Medium grid (10x10) - some scaling
            ((20, 20), True),  # Large grid (20x20) - significant scaling
            ((30, 30), True),  # Very large grid (30x30) - max scaling
        ],
    )
    def test_calculate_scale(self, grid_size, expected_scale_increase: bool):
        """Test grid scaling behavior with various grid sizes."""
        config = GridAutoscalerHyperparams()
        result = GridAutoscaler.compute(grid_size, config)

        # Check return type
        assert isinstance(result, GridScaleResult)

        # Check scale bounds
        assert result.figure_width_scale >= 1.0
        assert result.figure_height_scale >= 1.0
        assert result.font_size > 0
        assert result.annotation_font_size > 0

        if expected_scale_increase:
            assert result.figure_width_scale > 1.0
            assert result.figure_height_scale > 1.0

    def test_calculate_scale_with_custom_config(self):
        """Test GridAutoscaler with custom configuration."""
        custom_config = GridAutoscalerHyperparams(
            base_figure_width=9.0,
            base_figure_height=9.0,
            base_annotation_font_size=10.0,
            grid_cells_per_base_size=5,
        )

        result = GridAutoscaler.compute((7, 7), custom_config)

        # Should use custom configuration
        assert result.figure_width_scale <= custom_config.max_scale_factor
        assert result.annotation_font_size > 0

    def test_calculate_scale_with_none_config(self):
        """Test GridAutoscaler creates default config when None."""
        result = GridAutoscaler.compute((10, 10), None)

        assert isinstance(result, GridScaleResult)
        assert result.figure_width_scale >= 1.0
        assert result.figure_height_scale >= 1.0

    def test_calculate_scale_equal_scaling(self):
        """Test GridAutoscaler scales width and height equally."""
        config = GridAutoscalerHyperparams()
                result = GridAutoscaler.compute((20, 20), config)
        
        # For square grids, width and height should scale equally
        assert result.figure_width_scale == result.figure_height_scale
    
    def test_calculate_scale_edge_cases(self):
        """Test GridAutoscaler with edge cases."""
        config = GridAutoscalerHyperparams()
        
        # Zero grid size (treated as 1x1)
        result = GridAutoscaler.compute((1, 1), config)
        assert result.figure_width_scale >= 1.0
        assert result.figure_height_scale >= 1.0
        
        # Very large grid
        result = GridAutoscaler.compute((100, 100), config)
        assert result.figure_width_scale <= config.max_scale_factor
        assert result.figure_height_scale <= config.max_scale_factor
    
    def test_calculate_scale_consistency(self):
        """Test that scaling is consistent and predictable."""
        config = GridAutoscalerHyperparams()
        
        # Larger grids should have larger or equal scaling
        result_small = GridAutoscaler.compute((7, 7), config)
        result_large = GridAutoscaler.compute((14, 14), config)
        
        assert result_small.figure_width_scale <= result_large.figure_width_scale
        assert result_small.figure_height_scale <= result_large.figure_height_scale
    
    def test_calculate_scale_bounds_respected(self):
        """Test that scale factors respect min/max bounds."""
        config = GridAutoscalerHyperparams(min_scale_factor=0.3, max_scale_factor=2.5)
        
        # Test with extreme values
        result_small = GridAutoscaler.compute((1, 1), config)
        result_large = GridAutoscaler.compute((100, 100), config)

        # Scaling should respect max bounds
        assert result_large.figure_width_scale <= config.max_scale_factor
        assert result_large.figure_height_scale <= config.max_scale_factor

        # Font scaling should respect min bounds
        assert result_large.font_size >= config.base_font_size * config.min_scale_factor
        assert result_large.annotation_font_size >= (
            config.base_annotation_font_size * config.min_scale_factor
        )

    def test_calculate_scale_annotation_font_scaling(self):
        """Test that annotation font size scales appropriately."""
        config = GridAutoscalerHyperparams()

        # Test that annotation font size decreases with larger grids
        result_small = GridAutoscaler.compute(50, config)
        result_large = GridAutoscaler.compute(1000, config)

        # Annotation font should be smaller for larger grids
        assert result_large.annotation_font_size <= result_small.annotation_font_size

        # But should never be zero or negative
        assert result_large.annotation_font_size > 0

    def test_calculate_scale_square_assumption(self):
        """Test scaling assumes square grids for equal width/height scaling."""
        config = GridAutoscalerHyperparams()

        # Test various grid sizes
        test_sizes = [25, 100, 400, 900]

        for grid_size in test_sizes:
            result = GridAutoscaler.compute(grid_size, config)
            # Width and height should always scale equally
            assert result.figure_width_scale == result.figure_height_scale
