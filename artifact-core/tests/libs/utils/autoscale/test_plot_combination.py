"""Tests for PlotCombinationAutoscaler."""

import pytest
from artifact_core.libs.utils.autoscale.plot_combination import (
    PlotCombinationAutoscaler,
)
from artifact_core.libs.utils.autoscale.subplot import SubplotScaleConfig
from artifact_core.libs.utils.plot_combiner import PlotCombinationConfig


class TestPlotCombinationAutoscaler:
    """Test cases for PlotCombinationAutoscaler."""
    
    def test_create_autoscaled_config_basic(self):
        """Test basic PlotCombinationAutoscaler functionality."""
        base_config = PlotCombinationConfig(
            n_cols=3,
            figsize_horizontal_multiplier=6.0,
            figsize_vertical_multiplier=4.0,
            fig_title_fontsize=16.0,
            tight_layout_pad=0.5,
        )
        
        result = PlotCombinationAutoscaler.create_autoscaled_config(
            num_plots=12, base_config=base_config
        )
        
        # Check return type
        assert isinstance(result, PlotCombinationConfig)
        
        # Check that multipliers are scaled
        assert result.figsize_horizontal_multiplier != base_config.figsize_horizontal_multiplier
        assert result.figsize_vertical_multiplier != base_config.figsize_vertical_multiplier
    
    def test_create_autoscaled_config_with_plot_counts(self):
        """Test PlotCombinationAutoscaler with various plot counts."""
        base_config = PlotCombinationConfig(
            n_cols=3,
            figsize_horizontal_multiplier=6.0,
            figsize_vertical_multiplier=4.0,
        )
        
        # Test with different plot counts
        config_few = PlotCombinationAutoscaler.create_autoscaled_config(
            num_plots=4, base_config=base_config
        )
        config_many = PlotCombinationAutoscaler.create_autoscaled_config(
            num_plots=25, base_config=base_config
        )
        
        assert isinstance(config_few, PlotCombinationConfig)
        assert isinstance(config_many, PlotCombinationConfig)
        
        # Many plots should result in larger figure
        assert (config_many.figsize_horizontal_multiplier > 
                config_few.figsize_horizontal_multiplier)
        assert (config_many.figsize_vertical_multiplier > 
                config_few.figsize_vertical_multiplier)
    
    def test_create_autoscaled_config_with_none_config(self):
        """Test PlotCombinationAutoscaler with None subplot config."""
        base_config = PlotCombinationConfig(
            figsize_horizontal_multiplier=5.0,
            figsize_vertical_multiplier=5.0,
        )
        
        result = PlotCombinationAutoscaler.create_autoscaled_config(
            num_plots=10, base_config=base_config, subplot_config=None
        )
        
        assert isinstance(result, PlotCombinationConfig)
        # Should use default SubplotScaleConfig internally
    
    def test_create_autoscaled_config_with_custom_subplot_config(self):
        """Test PlotCombinationAutoscaler with custom subplot config."""
        base_config = PlotCombinationConfig(
            figsize_horizontal_multiplier=6.0,
            figsize_vertical_multiplier=4.0,
            fig_title_fontsize=16.0,
        )
        
        custom_subplot_config = SubplotScaleConfig(
            base_font_size=20.0,
            features_per_base_subplot=6
        )
        
        result = PlotCombinationAutoscaler.create_autoscaled_config(
            num_plots=12,
            base_config=base_config,
            subplot_config=custom_subplot_config
        )
        
        assert isinstance(result, PlotCombinationConfig)
        # Should use the custom subplot config for scaling
    
    def test_create_autoscaled_config_preserves_base_properties(self):
        """Test that base config properties are preserved."""
        base_config = PlotCombinationConfig(
            n_cols=2,
            dpi=300,
            tight_layout_rect=(0, 0, 1, 0.95),
            include_fig_titles=True,
            combined_title="Test Title",
            subplots_adjust_hspace=0.3,
            subplots_adjust_wspace=0.4,
            figsize_horizontal_multiplier=6.0,
            figsize_vertical_multiplier=4.0,
        )
        
        result = PlotCombinationAutoscaler.create_autoscaled_config(
            num_plots=12, base_config=base_config
        )
        
        # These should be preserved exactly
        assert result.n_cols == base_config.n_cols
        assert result.dpi == base_config.dpi
        assert result.tight_layout_rect == base_config.tight_layout_rect
        assert result.include_fig_titles == base_config.include_fig_titles
        assert result.combined_title == base_config.combined_title
        assert result.subplots_adjust_hspace == base_config.subplots_adjust_hspace
        assert result.subplots_adjust_wspace == base_config.subplots_adjust_wspace
    
    def test_create_autoscaled_config_scales_font_size(self):
        """Test that font size is scaled appropriately."""
        base_config = PlotCombinationConfig(
            figsize_horizontal_multiplier=6.0,
            figsize_vertical_multiplier=4.0,
            fig_title_fontsize=16.0,
        )
        
        result = PlotCombinationAutoscaler.create_autoscaled_config(
            num_plots=20, base_config=base_config
        )
        
        # Font size should be scaled
        assert result.fig_title_fontsize != base_config.fig_title_fontsize
        assert result.fig_title_fontsize > 0
    
    def test_create_autoscaled_config_scales_padding(self):
        """Test that padding is scaled appropriately."""
        base_config = PlotCombinationConfig(
            figsize_horizontal_multiplier=6.0,
            figsize_vertical_multiplier=4.0,
            tight_layout_pad=0.5,
        )
        
        result = PlotCombinationAutoscaler.create_autoscaled_config(
            num_plots=20, base_config=base_config
        )
        
        # Padding should be scaled
        assert result.tight_layout_pad != base_config.tight_layout_pad
        assert result.tight_layout_pad > 0
    
    def test_create_autoscaled_config_edge_cases(self):
        """Test PlotCombinationAutoscaler with edge cases."""
        base_config = PlotCombinationConfig(
            figsize_horizontal_multiplier=6.0,
            figsize_vertical_multiplier=4.0,
        )
        
        # Test with zero plots
        result_zero = PlotCombinationAutoscaler.create_autoscaled_config(
            num_plots=0, base_config=base_config
        )
        assert isinstance(result_zero, PlotCombinationConfig)
        
        # Test with one plot
        result_one = PlotCombinationAutoscaler.create_autoscaled_config(
            num_plots=1, base_config=base_config
        )
        assert isinstance(result_one, PlotCombinationConfig)
        
        # Test with very many plots
        result_many = PlotCombinationAutoscaler.create_autoscaled_config(
            num_plots=1000, base_config=base_config
        )
        assert isinstance(result_many, PlotCombinationConfig)
    
    def test_create_autoscaled_config_consistency(self):
        """Test that autoscaling is consistent and predictable."""
        base_config = PlotCombinationConfig(
            figsize_horizontal_multiplier=6.0,
            figsize_vertical_multiplier=4.0,
        )
        
        # More plots should result in larger or equal scaling
        result_few = PlotCombinationAutoscaler.create_autoscaled_config(
            num_plots=5, base_config=base_config
        )
        result_many = PlotCombinationAutoscaler.create_autoscaled_config(
            num_plots=20, base_config=base_config
        )
        
        assert (result_many.figsize_horizontal_multiplier >= 
                result_few.figsize_horizontal_multiplier)
        assert (result_many.figsize_vertical_multiplier >= 
                result_few.figsize_vertical_multiplier)
    
    def test_create_autoscaled_config_all_optional_params(self):
        """Test with all optional parameters in base config."""
        base_config = PlotCombinationConfig(
            n_cols=4,
            dpi=150,
            figsize_horizontal_multiplier=8.0,
            figsize_vertical_multiplier=6.0,
            tight_layout_rect=(0.05, 0.05, 0.95, 0.9),
            tight_layout_pad=1.0,
            subplots_adjust_hspace=0.2,
            subplots_adjust_wspace=0.3,
            include_fig_titles=False,
            fig_title_fontsize=18.0,
            combined_title="Combined Plot",
            combined_title_vertical_position=0.98,
        )
        
        result = PlotCombinationAutoscaler.create_autoscaled_config(
            num_plots=16, base_config=base_config
        )
        
        # Should handle all parameters without error
        assert isinstance(result, PlotCombinationConfig)
        assert result.n_cols == base_config.n_cols
        assert result.dpi == base_config.dpi
        assert result.combined_title == base_config.combined_title
        assert result.combined_title_vertical_position == base_config.combined_title_vertical_position 