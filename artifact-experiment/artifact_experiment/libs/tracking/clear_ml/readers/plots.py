from typing import Any, Dict, List

from artifact_experiment.libs.tracking.clear_ml.adapter import (
    ClearMLRunAdapter,
)
from artifact_experiment.libs.tracking.clear_ml.readers.base import ClearMLReader


class ClearMLPlotReader(ClearMLReader):
    @staticmethod
    def get_plot_path_from_metadata(dict_plot_metadata: Dict[str, Any]) -> str:
        return dict_plot_metadata["metric"]

    @staticmethod
    def get_plot_iter_from_metadata(dict_plot_metadata: Dict[str, Any]) -> int:
        return dict_plot_metadata["iter"]

    @classmethod
    def get_series_metadata(cls, run: ClearMLRunAdapter, plot_path: str) -> List[Dict[str, Any]]:
        plot_path = cls._get_full_path(path=plot_path)
        ls_all_plots = cls._get_all_plots(run)
        ls_series_metadata = cls._get_series_metadata(
            ls_all_plots=ls_all_plots, plot_path=plot_path
        )
        return ls_series_metadata

    @classmethod
    def _get_series_metadata(
        cls, ls_all_plots: List[Dict[str, Any]], plot_path: str
    ) -> List[Dict[str, Any]]:
        ls_series_metadata = [
            dict_plot_metadata
            for dict_plot_metadata in ls_all_plots
            if cls.get_plot_path_from_metadata(dict_plot_metadata=dict_plot_metadata) == plot_path
        ]
        return ls_series_metadata

    @classmethod
    def _get_all_plots(cls, run: ClearMLRunAdapter) -> List[Dict[str, Any]]:
        ls_all_plots = run.get_exported_plots()
        return ls_all_plots
