from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

import torch
from artifact_experiment.base.data_split import DataSplit
from artifact_experiment.base.tracking.client import TrackingClient
from matplotlib.figure import Figure
from numpy import ndarray
from tqdm import tqdm

from artifact_torch.base.components.callbacks.loader_hook import (
    DataLoaderHookArrayCallback,
    DataLoaderHookArrayCollectionCallback,
    DataLoaderHookArrayCollectionHandler,
    DataLoaderHookArrayHandler,
    DataLoaderHookCallback,
    DataLoaderHookCallbackHandler,
    DataLoaderHookCallbackResources,
    DataLoaderHookPlotCallback,
    DataLoaderHookPlotCollectionCallback,
    DataLoaderHookPlotCollectionHandler,
    DataLoaderHookPlotHandler,
    DataLoaderHookScoreCallback,
    DataLoaderHookScoreCollectionCallback,
    DataLoaderHookScoreCollectionHandler,
    DataLoaderHookScoreHandler,
)
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelTContr = TypeVar("ModelTContr", bound=Model, contravariant=True)
ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
DataLoaderHookRoutineT = TypeVar("DataLoaderHookRoutineT", bound="DataLoaderHookRoutine")


class DataLoaderHookRoutine(ABC, Generic[ModelTContr, ModelInputTContr, ModelOutputTContr]):
    _verbose = True
    _progressbar_message = "Processing Data Loader (Hooks)"

    def __init__(
        self,
        dict_data_loader_hook_handlers: Dict[
            str,
            DataLoaderHookCallbackHandler[
                Any, ModelTContr, ModelInputTContr, ModelOutputTContr, Any
            ],
        ],
        data_split: DataSplit,
    ):
        self._dict_handlers = dict_data_loader_hook_handlers
        self._data_split = data_split

    @classmethod
    def build(
        cls: Type[DataLoaderHookRoutineT],
        data_split: DataSplit,
        tracking_client: Optional[TrackingClient] = None,
    ) -> DataLoaderHookRoutineT:
        routine = cls._build(
            ls_score_callbacks=cls._get_score_callbacks(data_split=data_split),
            ls_array_callbacks=cls._get_array_callbacks(data_split=data_split),
            ls_plot_callbacks=cls._get_plot_callbacks(data_split=data_split),
            ls_score_collection_callbacks=cls._get_score_collection_callbacks(
                data_split=data_split
            ),
            ls_array_collection_callbacks=cls._get_array_collection_callbacks(
                data_split=data_split
            ),
            ls_plot_collection_callbacks=cls._get_plot_collection_callbacks(data_split=data_split),
            data_split=data_split,
            tracking_client=tracking_client,
        )
        return routine

    @property
    def scores(self) -> Dict[str, float]:
        if self._score_handler is not None:
            return self._score_handler.active_cache.copy()
        else:
            return {}

    @property
    def arrays(self) -> Dict[str, ndarray]:
        if self._array_handler is not None:
            return self._array_handler.active_cache.copy()
        else:
            return {}

    @property
    def plots(self) -> Dict[str, Figure]:
        if self._plot_handler is not None:
            return self._plot_handler.active_cache.copy()
        else:
            return {}

    @property
    def score_collections(self) -> Dict[str, Dict[str, float]]:
        if self._score_collection_handler is not None:
            return self._score_collection_handler.active_cache.copy()
        else:
            return {}

    @property
    def array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        if self._array_collection_handler is not None:
            return self._array_collection_handler.active_cache.copy()
        else:
            return {}

    @property
    def plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        if self._plot_collection_handler is not None:
            return self._plot_collection_handler.active_cache.copy()
        else:
            return {}

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
    ) -> List[
        DataLoaderHookCallbackHandler[
            DataLoaderHookCallback[ModelTContr, ModelInputTContr, ModelOutputTContr, Any, Any],
            ModelTContr,
            ModelInputTContr,
            ModelOutputTContr,
            Any,
        ]
    ]:
        return list(self._dict_handlers.values())

    @property
    def _score_handler(
        self,
    ) -> Optional[
        DataLoaderHookCallbackHandler[
            DataLoaderHookScoreCallback[ModelTContr, ModelInputTContr, ModelOutputTContr],
            ModelTContr,
            ModelInputTContr,
            ModelOutputTContr,
            float,
        ]
    ]:
        return self._dict_handlers.get("scores")

    @property
    def _array_handler(
        self,
    ) -> Optional[
        DataLoaderHookCallbackHandler[
            DataLoaderHookArrayCallback[ModelTContr, ModelInputTContr, ModelOutputTContr],
            ModelTContr,
            ModelInputTContr,
            ModelOutputTContr,
            ndarray,
        ]
    ]:
        return self._dict_handlers.get("arrays")

    @property
    def _plot_handler(
        self,
    ) -> Optional[
        DataLoaderHookCallbackHandler[
            DataLoaderHookPlotCallback[ModelTContr, ModelInputTContr, ModelOutputTContr],
            ModelTContr,
            ModelInputTContr,
            ModelOutputTContr,
            Figure,
        ]
    ]:
        return self._dict_handlers.get("plots")

    @property
    def _score_collection_handler(
        self,
    ) -> Optional[
        DataLoaderHookCallbackHandler[
            DataLoaderHookScoreCollectionCallback[ModelTContr, ModelInputTContr, ModelOutputTContr],
            ModelTContr,
            ModelInputTContr,
            ModelOutputTContr,
            Dict[str, float],
        ]
    ]:
        return self._dict_handlers.get("score_collections")

    @property
    def _array_collection_handler(
        self,
    ) -> Optional[
        DataLoaderHookCallbackHandler[
            DataLoaderHookArrayCollectionCallback[ModelTContr, ModelInputTContr, ModelOutputTContr],
            ModelTContr,
            ModelInputTContr,
            ModelOutputTContr,
            Dict[str, ndarray],
        ]
    ]:
        return self._dict_handlers.get("array_collections")

    @property
    def _plot_collection_handler(
        self,
    ) -> Optional[
        DataLoaderHookCallbackHandler[
            DataLoaderHookPlotCollectionCallback[ModelTContr, ModelInputTContr, ModelOutputTContr],
            ModelTContr,
            ModelInputTContr,
            ModelOutputTContr,
            Dict[str, Figure],
        ]
    ]:
        return self._dict_handlers.get("plot_collections")

    @staticmethod
    @abstractmethod
    def _get_score_callbacks(
        data_split: DataSplit,
    ) -> List[DataLoaderHookScoreCallback[ModelTContr, ModelInputTContr, ModelOutputTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_array_callbacks(
        data_split: DataSplit,
    ) -> List[DataLoaderHookArrayCallback[ModelTContr, ModelInputTContr, ModelOutputTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_callbacks(
        data_split: DataSplit,
    ) -> List[DataLoaderHookPlotCallback[ModelTContr, ModelInputTContr, ModelOutputTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_callbacks(
        data_split: DataSplit,
    ) -> List[
        DataLoaderHookScoreCollectionCallback[ModelTContr, ModelInputTContr, ModelOutputTContr]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_callbacks(
        data_split: DataSplit,
    ) -> List[
        DataLoaderHookArrayCollectionCallback[ModelTContr, ModelInputTContr, ModelOutputTContr]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_callbacks(
        data_split: DataSplit,
    ) -> List[
        DataLoaderHookPlotCollectionCallback[ModelTContr, ModelInputTContr, ModelOutputTContr]
    ]: ...

    def clear_cache(self):
        self._clear_cache()

    def should_trigger(self, n_epochs_elapsed: int) -> bool:
        return self._should_trigger(n_epochs_elapsed=n_epochs_elapsed)

    def attach_hooks(self, model: ModelTContr, n_epochs_elapsed: int):
        self._attach_hooks(model=model, n_epochs_elapsed=n_epochs_elapsed)

    def detach_hooks(self, n_epochs_elapsed: int):
        self._detach_hooks(n_epochs_elapsed=n_epochs_elapsed)

    def finalize(self, n_epochs_elapsed: int):
        self._finalize(n_epochs_elapsed=n_epochs_elapsed)

    def export(self):
        self._export()

    def execute(
        self, model: ModelTContr, data_loader: DataLoader[ModelInputTContr], n_epochs_elapsed: int
    ):
        resources = DataLoaderHookCallbackResources[ModelTContr, ModelInputTContr](
            step=n_epochs_elapsed, model=model, data_loader=data_loader
        )
        if self._has_callbacks:
            self._execute_parallel(resources=resources)

    def _execute_sequential(
        self,
        resources: DataLoaderHookCallbackResources[ModelTContr, ModelInputTContr],
    ):
        for handler in self._ls_handlers:
            handler.execute(resources=resources)

    def _execute_parallel(
        self,
        resources: DataLoaderHookCallbackResources[ModelTContr, ModelInputTContr],
    ):
        self._clear_cache()
        if self._should_trigger(n_epochs_elapsed=resources.step):
            self._attach_hooks(model=resources.model, n_epochs_elapsed=resources.step)
            resources.model.eval()
            try:
                with torch.no_grad():
                    for model_input in tqdm(
                        resources.data_loader,
                        desc=self._progressbar_message,
                        disable=not self._verbose,
                        leave=False,
                    ):
                        _ = resources.model(model_input)
            finally:
                self._detach_hooks(n_epochs_elapsed=resources.step)
            self._finalize(n_epochs_elapsed=resources.step)
            self._export()

    def _clear_cache(self):
        for loader_handler in self._ls_handlers:
            loader_handler.clear()

    def _should_trigger(self, n_epochs_elapsed: int) -> bool:
        ls_active_handlers = [
            handler
            for handler in self._ls_handlers
            if handler.should_trigger(n_epochs_elapsed=n_epochs_elapsed)
        ]
        return len(ls_active_handlers) > 0

    def _attach_hooks(self, model: ModelTContr, n_epochs_elapsed: int):
        for handler in self._ls_handlers:
            handler.attach_hooks(model=model, n_epochs_elapsed=n_epochs_elapsed)

    def _detach_hooks(self, n_epochs_elapsed: int):
        for handler in self._ls_handlers:
            handler.detach_hooks(n_epochs_elapsed=n_epochs_elapsed)

    def _finalize(self, n_epochs_elapsed: int):
        for handler in self._ls_handlers:
            handler.finalize(n_epochs_elapsed=n_epochs_elapsed)

    def _export(self):
        for handler in self._ls_handlers:
            handler.export()

    @classmethod
    def _build(
        cls: Type[DataLoaderHookRoutineT],
        ls_score_callbacks: List[
            DataLoaderHookScoreCallback[ModelTContr, ModelInputTContr, ModelOutputTContr]
        ],
        ls_array_callbacks: List[
            DataLoaderHookArrayCallback[ModelTContr, ModelInputTContr, ModelOutputTContr]
        ],
        ls_plot_callbacks: List[
            DataLoaderHookPlotCallback[ModelTContr, ModelInputTContr, ModelOutputTContr]
        ],
        ls_score_collection_callbacks: List[
            DataLoaderHookScoreCollectionCallback[ModelTContr, ModelInputTContr, ModelOutputTContr]
        ],
        ls_array_collection_callbacks: List[
            DataLoaderHookArrayCollectionCallback[ModelTContr, ModelInputTContr, ModelOutputTContr]
        ],
        ls_plot_collection_callbacks: List[
            DataLoaderHookPlotCollectionCallback[ModelTContr, ModelInputTContr, ModelOutputTContr]
        ],
        data_split: DataSplit,
        tracking_client: Optional[TrackingClient] = None,
    ) -> DataLoaderHookRoutineT:
        dict_data_loader_hook_handlers = cls._build_data_loader_hook_handlers(
            ls_score_callbacks=ls_score_callbacks,
            ls_array_callbacks=ls_array_callbacks,
            ls_plot_callbacks=ls_plot_callbacks,
            ls_score_collection_callbacks=ls_score_collection_callbacks,
            ls_array_collection_callbacks=ls_array_collection_callbacks,
            ls_plot_collection_callbacks=ls_plot_collection_callbacks,
            tracking_client=tracking_client,
        )
        routine = cls(
            dict_data_loader_hook_handlers=dict_data_loader_hook_handlers,
            data_split=data_split,
        )
        return routine

    @staticmethod
    def _build_data_loader_hook_handlers(
        ls_score_callbacks: List[
            DataLoaderHookScoreCallback[ModelTContr, ModelInputTContr, ModelOutputTContr]
        ],
        ls_array_callbacks: List[
            DataLoaderHookArrayCallback[ModelTContr, ModelInputTContr, ModelOutputTContr]
        ],
        ls_plot_callbacks: List[
            DataLoaderHookPlotCallback[ModelTContr, ModelInputTContr, ModelOutputTContr]
        ],
        ls_score_collection_callbacks: List[
            DataLoaderHookScoreCollectionCallback[ModelTContr, ModelInputTContr, ModelOutputTContr]
        ],
        ls_array_collection_callbacks: List[
            DataLoaderHookArrayCollectionCallback[ModelTContr, ModelInputTContr, ModelOutputTContr]
        ],
        ls_plot_collection_callbacks: List[
            DataLoaderHookPlotCollectionCallback[ModelTContr, ModelInputTContr, ModelOutputTContr]
        ],
        tracking_client: Optional[TrackingClient] = None,
    ) -> Dict[
        str,
        DataLoaderHookCallbackHandler[Any, ModelTContr, ModelInputTContr, ModelOutputTContr, Any],
    ]:
        dict_data_loader_hook_handlers = {
            "scores": DataLoaderHookScoreHandler[ModelTContr, ModelInputTContr, ModelOutputTContr](
                ls_callbacks=ls_score_callbacks,
                tracking_client=tracking_client,
            ),
            "arrays": DataLoaderHookArrayHandler[ModelTContr, ModelInputTContr, ModelOutputTContr](
                ls_callbacks=ls_array_callbacks,
                tracking_client=tracking_client,
            ),
            "plots": DataLoaderHookPlotHandler[ModelTContr, ModelInputTContr, ModelOutputTContr](
                ls_callbacks=ls_plot_callbacks,
                tracking_client=tracking_client,
            ),
            "score_collections": DataLoaderHookScoreCollectionHandler[
                ModelTContr, ModelInputTContr, ModelOutputTContr
            ](
                ls_callbacks=ls_score_collection_callbacks,
                tracking_client=tracking_client,
            ),
            "array_collections": DataLoaderHookArrayCollectionHandler[
                ModelTContr, ModelInputTContr, ModelOutputTContr
            ](
                ls_callbacks=ls_array_collection_callbacks,
                tracking_client=tracking_client,
            ),
            "plot_collections": DataLoaderHookPlotCollectionHandler[
                ModelTContr, ModelInputTContr, ModelOutputTContr
            ](
                ls_callbacks=ls_plot_collection_callbacks,
                tracking_client=tracking_client,
            ),
        }
        return dict_data_loader_hook_handlers
