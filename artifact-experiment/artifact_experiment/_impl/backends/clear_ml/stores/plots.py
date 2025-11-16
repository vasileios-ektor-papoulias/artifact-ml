from typing import Any, Dict, List, Type, TypeVar

from artifact_experiment._impl.backends.clear_ml.stores.store import ClearMLStore

ClearMLPlotStoreT = TypeVar("ClearMLPlotStoreT", bound="ClearMLPlotStore")


class ClearMLPlot:
    def __init__(self, dict_plot_metadata: Dict[str, Any]):
        self._metadata = dict_plot_metadata

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata.copy()

    @property
    def path(self) -> str:
        return self._metadata["metric"]

    @property
    def n_entries(self) -> int:
        return self._metadata["iter"]


ClearMLPlotSeries = List[ClearMLPlot]


class ClearMLPlotStore(ClearMLStore[ClearMLPlotSeries]):
    def __init__(self, root_dir: str, ls_all_plots: List[ClearMLPlot]):
        super().__init__(root_dir=root_dir)
        self._ls_all_plots = ls_all_plots

    @property
    def ls_all_plots(self) -> List[ClearMLPlot]:
        return self._ls_all_plots

    @property
    def n_entries(self) -> int:
        return len(self._ls_all_plots)

    @classmethod
    def build(
        cls: Type[ClearMLPlotStoreT],
        raw_plot_data: List[Dict[str, Any]],
        root_dir: str = "",
    ) -> ClearMLPlotStoreT:
        if root_dir is None:
            root_dir = ""
        ls_all_plots = [
            ClearMLPlot(dict_plot_metadata=dict_plot_metadata)
            for dict_plot_metadata in raw_plot_data
        ]
        ls_artifact_plots = [plot for plot in ls_all_plots if plot.path.startswith(root_dir)]
        store = cls(root_dir=root_dir, ls_all_plots=ls_artifact_plots)
        return store

    def __getitem__(self, idx: int):
        return self._ls_all_plots[idx]

    def _get(self, path: str) -> ClearMLPlotSeries:
        series = self._get_series(ls_all_plots=self._ls_all_plots, path=path)
        return series

    @classmethod
    def _get_series(cls, ls_all_plots: List[ClearMLPlot], path: str) -> ClearMLPlotSeries:
        ls_series_plots = [plot for plot in ls_all_plots if plot.path == path]
        return ls_series_plots
