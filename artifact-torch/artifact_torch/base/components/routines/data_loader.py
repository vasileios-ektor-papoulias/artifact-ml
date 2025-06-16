from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from artifact_experiment.base.tracking.client import TrackingClient
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_torch.base.components.callbacks.data_loader import (
    DataLoaderArrayCallback,
    DataLoaderArrayCollectionCallback,
    DataLoaderArrayCollectionHandler,
    DataLoaderArrayHandler,
    DataLoaderCallback,
    DataLoaderCallbackHandler,
    DataLoaderCallbackResources,
    DataLoaderPlotCallback,
    DataLoaderPlotCollectionCallback,
    DataLoaderPlotCollectionHandler,
    DataLoaderPlotHandler,
    DataLoaderScoreCallback,
    DataLoaderScoreCollectionCallback,
    DataLoaderScoreCollectionHandler,
    DataLoaderScoreHandler,
)
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
DataLoaderRoutineT = TypeVar("DataLoaderRoutineT", bound="DataLoaderRoutine")


class DataLoaderRoutine(ABC, Generic[ModelInputTContr, ModelOutputTContr]):
    def __init__(
        self,
        data_loader: DataLoader[ModelInputTContr],
        dict_data_loader_handlers: Dict[
            str, DataLoaderCallbackHandler[Any, ModelInputTContr, ModelOutputTContr, Any]
        ],
    ):
        self._data_loader = data_loader
        self._dict_handlers = dict_data_loader_handlers

    @classmethod
    def build(
        cls: Type[DataLoaderRoutineT],
        data_loader: DataLoader[ModelInputTContr],
        tracking_client: Optional[TrackingClient] = None,
    ) -> DataLoaderRoutineT:
        dict_data_loader_handlers = cls._build_data_loader_handlers(
            ls_score_callbacks=cls._get_score_callbacks(),
            ls_array_callbacks=cls._get_array_callbacks(),
            ls_plot_callbacks=cls._get_plot_callbacks(),
            ls_score_collection_callbacks=cls._get_score_collection_callbacks(),
            ls_array_collection_callbacks=cls._get_array_collection_callbacks(),
            ls_plot_collection_callbacks=cls._get_plot_collection_callbacks(),
            tracking_client=tracking_client,
        )
        routine = cls(
            data_loader=data_loader,
            dict_data_loader_handlers=dict_data_loader_handlers,
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
    def _ls_handlers(
        self,
    ) -> List[
        DataLoaderCallbackHandler[
            DataLoaderCallback[ModelInputTContr, ModelOutputTContr, Any, Any],
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
        DataLoaderCallbackHandler[
            DataLoaderScoreCallback[ModelInputTContr, ModelOutputTContr],
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
        DataLoaderCallbackHandler[
            DataLoaderArrayCallback[ModelInputTContr, ModelOutputTContr],
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
        DataLoaderCallbackHandler[
            DataLoaderPlotCallback[ModelInputTContr, ModelOutputTContr],
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
        DataLoaderCallbackHandler[
            DataLoaderScoreCollectionCallback[ModelInputTContr, ModelOutputTContr],
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
        DataLoaderCallbackHandler[
            DataLoaderArrayCallback[ModelInputTContr, ModelOutputTContr],
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
        DataLoaderCallbackHandler[
            DataLoaderPlotCallback[ModelInputTContr, ModelOutputTContr],
            ModelInputTContr,
            ModelOutputTContr,
            Dict[str, Figure],
        ]
    ]:
        return self._dict_handlers.get("plot_collecitons")

    @staticmethod
    @abstractmethod
    def _get_score_callbacks() -> List[
        DataLoaderScoreCallback[ModelInputTContr, ModelOutputTContr]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_array_callbacks() -> List[
        DataLoaderArrayCallback[ModelInputTContr, ModelOutputTContr]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_callbacks() -> List[
        DataLoaderPlotCallback[ModelInputTContr, ModelOutputTContr]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_callbacks() -> List[
        DataLoaderScoreCollectionCallback[ModelInputTContr, ModelOutputTContr]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_callbacks() -> List[
        DataLoaderArrayCollectionCallback[ModelInputTContr, ModelOutputTContr]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_callbacks() -> List[
        DataLoaderPlotCollectionCallback[ModelInputTContr, ModelOutputTContr]
    ]: ...

    def execute(self, model: Model[ModelInputTContr, ModelOutputTContr], n_epochs_elapsed: int):
        resources = DataLoaderCallbackResources[ModelInputTContr, ModelOutputTContr](
            step=n_epochs_elapsed, model=model, data_loader=self._data_loader
        )
        for loader_handler in self._ls_handlers:
            loader_handler.execute(resources=resources)

    def clear_cache(self):
        for loader_handler in self._ls_handlers:
            loader_handler.clear()

    @classmethod
    def _build(
        cls: Type[DataLoaderRoutineT],
        data_loader: DataLoader[ModelInputTContr],
        ls_score_callbacks: List[DataLoaderScoreCallback[ModelInputTContr, ModelOutputTContr]],
        ls_array_callbacks: List[DataLoaderArrayCallback[ModelInputTContr, ModelOutputTContr]],
        ls_plot_callbacks: List[DataLoaderPlotCallback[ModelInputTContr, ModelOutputTContr]],
        ls_score_collection_callbacks: List[
            DataLoaderScoreCollectionCallback[ModelInputTContr, ModelOutputTContr]
        ],
        ls_array_collection_callbacks: List[
            DataLoaderArrayCollectionCallback[ModelInputTContr, ModelOutputTContr]
        ],
        ls_plot_collection_callbacks: List[
            DataLoaderPlotCollectionCallback[ModelInputTContr, ModelOutputTContr]
        ],
        tracking_client: Optional[TrackingClient] = None,
    ) -> DataLoaderRoutineT:
        dict_data_loader_handlers = cls._build_data_loader_handlers(
            ls_score_callbacks=ls_score_callbacks,
            ls_array_callbacks=ls_array_callbacks,
            ls_plot_callbacks=ls_plot_callbacks,
            ls_score_collection_callbacks=ls_score_collection_callbacks,
            ls_array_collection_callbacks=ls_array_collection_callbacks,
            ls_plot_collection_callbacks=ls_plot_collection_callbacks,
            tracking_client=tracking_client,
        )
        routine = cls(
            data_loader=data_loader,
            dict_data_loader_handlers=dict_data_loader_handlers,
        )
        return routine

    @staticmethod
    def _build_data_loader_handlers(
        ls_score_callbacks: List[DataLoaderScoreCallback[ModelInputTContr, ModelOutputTContr]],
        ls_array_callbacks: List[DataLoaderArrayCallback[ModelInputTContr, ModelOutputTContr]],
        ls_plot_callbacks: List[DataLoaderPlotCallback[ModelInputTContr, ModelOutputTContr]],
        ls_score_collection_callbacks: List[
            DataLoaderScoreCollectionCallback[ModelInputTContr, ModelOutputTContr]
        ],
        ls_array_collection_callbacks: List[
            DataLoaderArrayCollectionCallback[ModelInputTContr, ModelOutputTContr]
        ],
        ls_plot_collection_callbacks: List[
            DataLoaderPlotCollectionCallback[ModelInputTContr, ModelOutputTContr]
        ],
        tracking_client: Optional[TrackingClient] = None,
    ) -> Dict[str, DataLoaderCallbackHandler[Any, ModelInputTContr, ModelOutputTContr, Any]]:
        dict_data_loader_handlers = {
            "scores": DataLoaderScoreHandler[ModelInputTContr, ModelOutputTContr](
                ls_callbacks=ls_score_callbacks,
                tracking_client=tracking_client,
            ),
            "arrays": DataLoaderArrayHandler[ModelInputTContr, ModelOutputTContr](
                ls_callbacks=ls_array_callbacks,
                tracking_client=tracking_client,
            ),
            "plots": DataLoaderPlotHandler[ModelInputTContr, ModelOutputTContr](
                ls_callbacks=ls_plot_callbacks, tracking_client=tracking_client
            ),
            "score_collections": DataLoaderScoreCollectionHandler[
                ModelInputTContr, ModelOutputTContr
            ](
                ls_callbacks=ls_score_collection_callbacks,
                tracking_client=tracking_client,
            ),
            "array_collections": DataLoaderArrayCollectionHandler[
                ModelInputTContr, ModelOutputTContr
            ](
                ls_callbacks=ls_array_collection_callbacks,
                tracking_client=tracking_client,
            ),
            "plot_collections": DataLoaderPlotCollectionHandler[
                ModelInputTContr, ModelOutputTContr
            ](
                ls_callbacks=ls_plot_collection_callbacks,
                tracking_client=tracking_client,
            ),
        }
        return dict_data_loader_handlers
