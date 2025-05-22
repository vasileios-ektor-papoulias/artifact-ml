from typing import Any, Dict, List

from artifact_experiment.libs.tracking.clear_ml.adapter import (
    ClearMLRunAdapter,
)


class ClearMLPlotReader:
    @classmethod
    def get_all_plots(cls, run: ClearMLRunAdapter) -> List[Dict[str, Any]]:
        ls_all_plots = run.get_exported_plots()
        return ls_all_plots

    @classmethod
    def get_series_from_path(
        cls, ls_all_plots: List[Dict[str, Any]], plot_path: str
    ) -> List[Dict[str, Any]]:
        ls_series_metadata = [
            dict_plot_metadata
            for dict_plot_metadata in ls_all_plots
            if cls.get_plot_path_from_metadata(dict_plot_metadata=dict_plot_metadata) == plot_path
        ]
        return ls_series_metadata

    @staticmethod
    def get_plot_path_from_metadata(dict_plot_metadata: Dict[str, Any]) -> str:
        return dict_plot_metadata["metric"]

    @staticmethod
    def get_plot_iter_from_metadata(dict_plot_metadata: Dict[str, Any]) -> int:
        return dict_plot_metadata["iter"]
