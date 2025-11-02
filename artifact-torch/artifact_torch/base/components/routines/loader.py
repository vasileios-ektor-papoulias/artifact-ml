from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

import torch
from artifact_experiment.base.data_split import DataSplit
from artifact_experiment.base.tracking.client import TrackingClient
from matplotlib.figure import Figure
from numpy import ndarray
from tqdm import tqdm

from artifact_torch.base.components.callbacks.forward_hook import (
    ForwardHookArray,
    ForwardHookArrayCollection,
    ForwardHookArrayCollectionHandler,
    ForwardHookArrayHandler,
    ForwardHookCallbackHandler,
    ForwardHookCallbackResources,
    ForwardHookHandlers,
    ForwardHookPlot,
    ForwardHookPlotCollection,
    ForwardHookPlotCollectionHandler,
    ForwardHookPlotHandler,
    ForwardHookScore,
    ForwardHookScoreCollection,
    ForwardHookScoreCollectionHandler,
    ForwardHookScoreHandler,
)
from artifact_torch.base.components.callbacks.loader import (
    DataLoaderArray,
    DataLoaderArrayCollection,
    DataLoaderArrayCollectionHandler,
    DataLoaderArrayHandler,
    DataLoaderHandlers,
    DataLoaderPlot,
    DataLoaderPlotCollection,
    DataLoaderPlotCollectionHandler,
    DataLoaderPlotHandler,
    DataLoaderScore,
    DataLoaderScoreCollection,
    DataLoaderScoreCollectionHandler,
    DataLoaderScoreHandler,
)
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelTContr = TypeVar("ModelTContr", bound=Model, contravariant=True)
ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
DataLoaderRoutineT = TypeVar("DataLoaderRoutineT", bound="DataLoaderRoutine")


class DataLoaderRoutine(ABC, Generic[ModelTContr, ModelInputTContr, ModelOutputTContr]):
    _verbose = True
    _progressbar_message = "Processing Data Loader"

    def __init__(
        self,
        loader_handlers: DataLoaderHandlers[ModelInputTContr, ModelOutputTContr],
        hook_handlers: ForwardHookHandlers[ModelTContr],
        data_loader: DataLoader[ModelInputTContr],
        data_split: DataSplit,
    ):
        self._loader_handlers = loader_handlers
        self._hook_handlers = hook_handlers
        self._data_loader = data_loader
        self._data_split = data_split

    @classmethod
    def build(
        cls: Type[DataLoaderRoutineT],
        data_loader: DataLoader[ModelInputTContr],
        data_split: DataSplit,
        tracking_client: Optional[TrackingClient] = None,
    ) -> DataLoaderRoutineT:
        routine = cls._build(
            ls_scores=cls._get_scores(data_split=data_split),
            ls_arrays=cls._get_arrays(data_split=data_split),
            ls_plots=cls._get_plots(data_split=data_split),
            ls_score_collections=cls._get_score_collections(data_split=data_split),
            ls_array_collections=cls._get_array_collections(data_split=data_split),
            ls_plot_collections=cls._get_plot_collections(data_split=data_split),
            ls_score_hooks=cls._get_score_hooks(data_split=data_split),
            ls_array_hooks=cls._get_array_hooks(data_split=data_split),
            ls_plot_hooks=cls._get_plot_hooks(data_split=data_split),
            ls_score_collection_hooks=cls._get_score_collection_hooks(data_split=data_split),
            ls_array_collection_hooks=cls._get_array_collection_hooks(data_split=data_split),
            ls_plot_collection_hooks=cls._get_plot_collection_hooks(data_split=data_split),
            data_loader=data_loader,
            data_split=data_split,
            tracking_client=tracking_client,
        )
        return routine

    @property
    def scores(self) -> Dict[str, float]:
        scores = {}
        scores.update(self._loader_score_handler.active_cache)
        scores.update(self._hook_score_handler.active_cache)
        return scores

    @property
    def arrays(self) -> Dict[str, ndarray]:
        arrays = {}
        arrays.update(self._loader_array_handler.active_cache)
        arrays.update(self._hook_array_handler.active_cache)
        return arrays

    @property
    def plots(self) -> Dict[str, Figure]:
        plots = {}
        plots.update(self._loader_plot_handler.active_cache)
        plots.update(self._hook_plot_handler.active_cache)
        return plots

    @property
    def score_collections(self) -> Dict[str, Dict[str, float]]:
        score_collections = {}
        score_collections.update(self._loader_score_collection_handler.active_cache)
        score_collections.update(self._hook_score_collection_handler.active_cache)
        return score_collections

    @property
    def array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        array_collections = {}
        array_collections.update(self._loader_array_collection_handler.active_cache)
        array_collections.update(self._hook_array_collection_handler.active_cache)
        return array_collections

    @property
    def plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        plot_collections = {}
        plot_collections.update(self._loader_plot_collection_handler.active_cache)
        plot_collections.update(self._hook_plot_collection_handler.active_cache)
        return plot_collections

    @property
    def data_split(self) -> DataSplit:
        return self._data_split

    @property
    def n_callbacks(self) -> int:
        return self._n_callbacks

    @property
    def has_callbacks(self) -> bool:
        return self._has_callbacks

    @property
    def _n_callbacks(self) -> int:
        return sum([handler.n_callbacks for handler in self._ls_handlers])

    @property
    def _has_callbacks(self) -> bool:
        return self._n_callbacks != 0

    @property
    def _ls_handlers(
        self,
    ) -> List[ForwardHookCallbackHandler[Any, Any, Any]]:
        return [
            self._loader_handlers.score_handler,
            self._loader_handlers.array_handler,
            self._loader_handlers.plot_handler,
            self._loader_handlers.score_collection_handler,
            self._loader_handlers.array_collection_handler,
            self._loader_handlers.plot_collection_handler,
            self._hook_handlers.score_handler,
            self._hook_handlers.array_handler,
            self._hook_handlers.plot_handler,
            self._hook_handlers.score_collection_handler,
            self._hook_handlers.array_collection_handler,
            self._hook_handlers.plot_collection_handler,
        ]

    @property
    def _ls_loader_handlers(
        self,
    ) -> Sequence[ForwardHookCallbackHandler[Any, Any, Any]]:
        return [
            self._loader_handlers.score_handler,
            self._loader_handlers.array_handler,
            self._loader_handlers.plot_handler,
            self._loader_handlers.score_collection_handler,
            self._loader_handlers.array_collection_handler,
            self._loader_handlers.plot_collection_handler,
        ]

    @property
    def _loader_score_handler(
        self,
    ) -> DataLoaderScoreHandler[ModelInputTContr, ModelOutputTContr]:
        return self._loader_handlers.score_handler

    @property
    def _loader_array_handler(
        self,
    ) -> DataLoaderArrayHandler[ModelInputTContr, ModelOutputTContr]:
        return self._loader_handlers.array_handler

    @property
    def _loader_plot_handler(
        self,
    ) -> DataLoaderPlotHandler[ModelInputTContr, ModelOutputTContr]:
        return self._loader_handlers.plot_handler

    @property
    def _loader_score_collection_handler(
        self,
    ) -> DataLoaderScoreCollectionHandler[ModelInputTContr, ModelOutputTContr]:
        return self._loader_handlers.score_collection_handler

    @property
    def _loader_array_collection_handler(
        self,
    ) -> DataLoaderArrayCollectionHandler[ModelInputTContr, ModelOutputTContr]:
        return self._loader_handlers.array_collection_handler

    @property
    def _loader_plot_collection_handler(
        self,
    ) -> DataLoaderPlotCollectionHandler[ModelInputTContr, ModelOutputTContr]:
        return self._loader_handlers.plot_collection_handler

    @property
    def _ls_hook_handlers(
        self,
    ) -> List[ForwardHookCallbackHandler[Any, Any, Any]]:
        return [
            self._hook_handlers.score_handler,
            self._hook_handlers.array_handler,
            self._hook_handlers.plot_handler,
            self._hook_handlers.score_collection_handler,
            self._hook_handlers.array_collection_handler,
            self._hook_handlers.plot_collection_handler,
        ]

    @property
    def _hook_score_handler(
        self,
    ) -> ForwardHookScoreHandler[ModelTContr]:
        return self._hook_handlers.score_handler

    @property
    def _hook_array_handler(
        self,
    ) -> ForwardHookArrayHandler[ModelTContr]:
        return self._hook_handlers.array_handler

    @property
    def _hook_plot_handler(
        self,
    ) -> ForwardHookPlotHandler[ModelTContr]:
        return self._hook_handlers.plot_handler

    @property
    def _hook_score_collection_handler(
        self,
    ) -> ForwardHookScoreCollectionHandler[ModelTContr]:
        return self._hook_handlers.score_collection_handler

    @property
    def _hook_array_collection_handler(
        self,
    ) -> ForwardHookArrayCollectionHandler[ModelTContr]:
        return self._hook_handlers.array_collection_handler

    @property
    def _hook_plot_collection_handler(
        self,
    ) -> ForwardHookPlotCollectionHandler[ModelTContr]:
        return self._hook_handlers.plot_collection_handler

    @staticmethod
    @abstractmethod
    def _get_scores(
        data_split: DataSplit,
    ) -> List[DataLoaderScore[ModelInputTContr, ModelOutputTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_arrays(
        data_split: DataSplit,
    ) -> List[DataLoaderArray[ModelInputTContr, ModelOutputTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_plots(
        data_split: DataSplit,
    ) -> List[DataLoaderPlot[ModelInputTContr, ModelOutputTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collections(
        data_split: DataSplit,
    ) -> List[DataLoaderScoreCollection[ModelInputTContr, ModelOutputTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collections(
        data_split: DataSplit,
    ) -> List[DataLoaderArrayCollection[ModelInputTContr, ModelOutputTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collections(
        data_split: DataSplit,
    ) -> List[DataLoaderPlotCollection[ModelInputTContr, ModelOutputTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_score_hooks(data_split: DataSplit) -> List[ForwardHookScore[ModelTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_array_hooks(data_split: DataSplit) -> List[ForwardHookArray[ModelTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_hooks(data_split: DataSplit) -> List[ForwardHookPlot[ModelTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_hooks(
        data_split: DataSplit,
    ) -> List[ForwardHookScoreCollection[ModelTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_hooks(
        data_split: DataSplit,
    ) -> List[ForwardHookArrayCollection[ModelTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_hooks(
        data_split: DataSplit,
    ) -> List[ForwardHookPlotCollection[ModelTContr]]: ...

    def clear_cache(self):
        for loader_handler in self._ls_handlers:
            loader_handler.clear()

    def execute(self, model: ModelTContr, n_epochs_elapsed: int):
        resources = ForwardHookCallbackResources[ModelTContr](step=n_epochs_elapsed, model=model)
        any_attached = False
        for handler in self._ls_handlers:
            any_attached |= handler.attach(resources=resources)
        if any_attached:
            self._process_data_loader(model=resources.model, data_loader=self._data_loader)
        for handler in self._ls_handlers:
            handler.execute(resources=resources)

    @classmethod
    def _process_data_loader(cls, model: ModelTContr, data_loader: DataLoader[ModelInputTContr]):
        model.eval()
        with torch.no_grad():
            for model_input in tqdm(
                data_loader,
                desc=cls._progressbar_message,
                disable=not cls._verbose,
                leave=False,
            ):
                _ = model(model_input)

    @classmethod
    def _build(
        cls: Type[DataLoaderRoutineT],
        ls_scores: List[DataLoaderScore[ModelInputTContr, ModelOutputTContr]],
        ls_arrays: List[DataLoaderArray[ModelInputTContr, ModelOutputTContr]],
        ls_plots: List[DataLoaderPlot[ModelInputTContr, ModelOutputTContr]],
        ls_score_collections: List[DataLoaderScoreCollection[ModelInputTContr, ModelOutputTContr]],
        ls_array_collections: List[DataLoaderArrayCollection[ModelInputTContr, ModelOutputTContr]],
        ls_plot_collections: List[DataLoaderPlotCollection[ModelInputTContr, ModelOutputTContr]],
        ls_score_hooks: List[ForwardHookScore[ModelTContr]],
        ls_array_hooks: List[ForwardHookArray[ModelTContr]],
        ls_plot_hooks: List[ForwardHookPlot[ModelTContr]],
        ls_score_collection_hooks: List[ForwardHookScoreCollection[ModelTContr]],
        ls_array_collection_hooks: List[ForwardHookArrayCollection[ModelTContr]],
        ls_plot_collection_hooks: List[ForwardHookPlotCollection[ModelTContr]],
        data_loader: DataLoader[ModelInputTContr],
        data_split: DataSplit,
        tracking_client: Optional[TrackingClient] = None,
    ) -> DataLoaderRoutineT:
        loader_handlers = DataLoaderHandlers.build(
            ls_score_callbacks=ls_scores,
            ls_array_callbacks=ls_arrays,
            ls_plot_callbacks=ls_plots,
            ls_score_collection_callbacks=ls_score_collections,
            ls_array_collection_callbacks=ls_array_collections,
            ls_plot_collection_callbacks=ls_plot_collections,
            tracking_client=tracking_client,
        )
        hook_handlers = ForwardHookHandlers.build(
            ls_score_callbacks=ls_score_hooks,
            ls_array_callbacks=ls_array_hooks,
            ls_plot_callbacks=ls_plot_hooks,
            ls_score_collection_callbacks=ls_score_collection_hooks,
            ls_array_collection_callbacks=ls_array_collection_hooks,
            ls_plot_collection_callbacks=ls_plot_collection_hooks,
            tracking_client=tracking_client,
        )
        routine = cls(
            data_loader=data_loader,
            loader_handlers=loader_handlers,
            hook_handlers=hook_handlers,
            data_split=data_split,
        )
        return routine
