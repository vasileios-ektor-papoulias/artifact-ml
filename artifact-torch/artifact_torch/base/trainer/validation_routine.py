from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
)
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.base.validation_plan import ValidationPlan
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_torch.base.components.callbacks.data_loader import (
    DataLoaderArrayCallback,
    DataLoaderArrayCollectionCallback,
    DataLoaderArrayCollectionHandler,
    DataLoaderArrayHandler,
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
from artifact_torch.base.components.callbacks.validation_plan import (
    ValidationPlanCallback,
    ValidationPlanCallbackResources,
)
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelInputT = TypeVar("ModelInputT", bound=ModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=ModelOutput)
ModelT = TypeVar("ModelT", bound=Model[Any, Any])
ValidationPlanT = TypeVar("ValidationPlanT", bound=ValidationPlan)
ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)
ValidationRoutineT = TypeVar("ValidationRoutineT", bound="ValidationRoutine")


class ValidationRoutine(
    Generic[ModelT, ModelInputT, ModelOutputT, ValidationPlanT, ArtifactResourcesT]
):
    def __init__(
        self,
        train_loader: DataLoader[ModelInputT],
        val_loader: Optional[DataLoader[ModelInputT]],
        validation_plan_callback: ValidationPlanCallback[
            ModelT, ValidationPlanT, ArtifactResourcesT
        ],
        train_loader_score_handler: DataLoaderScoreHandler[ModelInputT, ModelOutputT],
        train_loader_array_handler: DataLoaderArrayHandler[ModelInputT, ModelOutputT],
        train_loader_plot_handler: DataLoaderPlotHandler[ModelInputT, ModelOutputT],
        train_loader_score_collection_handler: DataLoaderScoreCollectionHandler[
            ModelInputT, ModelOutputT
        ],
        train_loader_array_collection_handler: DataLoaderArrayCollectionHandler[
            ModelInputT, ModelOutputT
        ],
        train_loader_plot_collection_handler: DataLoaderPlotCollectionHandler[
            ModelInputT, ModelOutputT
        ],
        val_loader_score_handler: DataLoaderScoreHandler[ModelInputT, ModelOutputT],
        val_loader_array_handler: DataLoaderArrayHandler[ModelInputT, ModelOutputT],
        val_loader_plot_handler: DataLoaderPlotHandler[ModelInputT, ModelOutputT],
        val_loader_score_collection_handler: DataLoaderScoreCollectionHandler[
            ModelInputT, ModelOutputT
        ],
        val_loader_array_collection_handler: DataLoaderArrayCollectionHandler[
            ModelInputT, ModelOutputT
        ],
        val_loader_plot_collection_handler: DataLoaderPlotCollectionHandler[
            ModelInputT, ModelOutputT
        ],
    ):
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._validation_plan_callback = validation_plan_callback
        self._train_loader_score_handler = train_loader_score_handler
        self._train_loader_array_handler = train_loader_array_handler
        self._train_loader_plot_handler = train_loader_plot_handler
        self._train_loader_score_collection_handler = train_loader_score_collection_handler
        self._train_loader_array_collection_handler = train_loader_array_collection_handler
        self._train_loader_plot_collection_handler = train_loader_plot_collection_handler
        self._val_loader_score_handler = val_loader_score_handler
        self._val_loader_array_handler = val_loader_array_handler
        self._val_loader_plot_handler = val_loader_plot_handler
        self._val_loader_score_collection_handler = val_loader_score_collection_handler
        self._val_loader_array_collection_handler = val_loader_array_collection_handler
        self._val_loader_plot_collection_handler = val_loader_plot_collection_handler

    @classmethod
    def _build(
        cls: Type[ValidationRoutineT],
        train_loader: DataLoader[ModelInputT],
        val_loader: Optional[DataLoader[ModelInputT]],
        validation_plan_callback: ValidationPlanCallback[
            ModelT, ValidationPlanT, ArtifactResourcesT
        ],
        tracking_client: Optional[TrackingClient],
    ) -> ValidationRoutineT:
        train_loader_score_callbacks = cls._get_train_loader_score_callbacks()
        train_loader_score_handler = DataLoaderScoreHandler[ModelInputT, ModelOutputT](
            ls_callbacks=train_loader_score_callbacks, tracking_client=tracking_client
        )
        train_loader_array_callbacks = cls._get_train_loader_array_callbacks()
        train_loader_array_handler = DataLoaderArrayHandler[ModelInputT, ModelOutputT](
            ls_callbacks=train_loader_array_callbacks
        )
        train_loader_plot_callbacks = cls._get_train_loader_plot_callbacks()
        train_loader_plot_handler = DataLoaderPlotHandler[ModelInputT, ModelOutputT](
            ls_callbacks=train_loader_plot_callbacks
        )
        train_loader_score_collection_callbacks = cls._get_train_loader_score_collection_callbacks()
        train_loader_score_collection_handler = DataLoaderScoreCollectionHandler[
            ModelInputT, ModelOutputT
        ](ls_callbacks=train_loader_score_collection_callbacks)
        train_loader_array_collection_callbacks = cls._get_train_loader_array_collection_callbacks()
        train_loader_array_collection_handler = DataLoaderArrayCollectionHandler[
            ModelInputT, ModelOutputT
        ](ls_callbacks=train_loader_array_collection_callbacks)
        train_loader_plot_collection_callbacks = cls._get_train_loader_plot_collection_callbacks()
        train_loader_plot_collection_handler = DataLoaderPlotCollectionHandler[
            ModelInputT, ModelOutputT
        ](ls_callbacks=train_loader_plot_collection_callbacks)
        val_loader_score_callbacks = cls._get_val_loader_score_callbacks()
        val_loader_score_handler = DataLoaderScoreHandler[ModelInputT, ModelOutputT](
            ls_callbacks=val_loader_score_callbacks
        )
        val_loader_array_callbacks = cls._get_val_loader_array_callbacks()
        val_loader_array_handler = DataLoaderArrayHandler[ModelInputT, ModelOutputT](
            ls_callbacks=val_loader_array_callbacks
        )
        val_loader_plot_callbacks = cls._get_val_loader_plot_callbacks()
        val_loader_plot_handler = DataLoaderPlotHandler[ModelInputT, ModelOutputT](
            ls_callbacks=val_loader_plot_callbacks
        )
        val_loader_score_collection_callbacks = cls._get_val_loader_score_collection_callbacks()
        val_loader_score_collection_handler = DataLoaderScoreCollectionHandler[
            ModelInputT, ModelOutputT
        ](ls_callbacks=val_loader_score_collection_callbacks)
        val_loader_array_collection_callbacks = cls._get_val_loader_array_collection_callbacks()
        val_loader_array_collection_handler = DataLoaderArrayCollectionHandler[
            ModelInputT, ModelOutputT
        ](ls_callbacks=val_loader_array_collection_callbacks)
        val_loader_plot_collection_callbacks = cls._get_val_loader_plot_collection_callbacks()
        val_loader_plot_collection_handler = DataLoaderPlotCollectionHandler[
            ModelInputT, ModelOutputT
        ](ls_callbacks=val_loader_plot_collection_callbacks)
        routine = cls(
            train_loader=train_loader,
            val_loader=val_loader,
            validation_plan_callback=validation_plan_callback,
            train_loader_score_handler=train_loader_score_handler,
            train_loader_array_handler=train_loader_array_handler,
            train_loader_plot_handler=train_loader_plot_handler,
            train_loader_score_collection_handler=train_loader_score_collection_handler,
            train_loader_array_collection_handler=train_loader_array_collection_handler,
            train_loader_plot_collection_handler=train_loader_plot_collection_handler,
            val_loader_score_handler=val_loader_score_handler,
            val_loader_array_handler=val_loader_array_handler,
            val_loader_plot_handler=val_loader_plot_handler,
            val_loader_score_collection_handler=val_loader_score_collection_handler,
            val_loader_array_collection_handler=val_loader_array_collection_handler,
            val_loader_plot_collection_handler=val_loader_plot_collection_handler,
        )
        return routine

    @property
    def scores(self) -> Dict[str, float]:
        scores = {}
        scores.update(self._validation_plan_callback.validation_plan.scores)
        scores.update(self._train_loader_score_handler.active_cache)
        scores.update(self._val_loader_score_handler.active_cache)
        return scores

    @property
    def arrays(self) -> Dict[str, ndarray]:
        arrays = {}
        arrays.update(self._validation_plan_callback.validation_plan.arrays)
        arrays.update(self._train_loader_array_handler.active_cache)
        arrays.update(self._val_loader_array_handler.active_cache)
        return arrays

    @property
    def plots(self) -> Dict[str, Figure]:
        plots = {}
        plots.update(self._validation_plan_callback.validation_plan.plots)
        plots.update(self._train_loader_plot_handler.active_cache)
        plots.update(self._val_loader_plot_handler.active_cache)
        return plots

    @property
    def score_collections(self) -> Dict[str, Dict[str, float]]:
        score_collections = {}
        score_collections.update(self._validation_plan_callback.validation_plan.score_collections)
        score_collections.update(self._train_loader_score_collection_handler.active_cache)
        score_collections.update(self._val_loader_score_collection_handler.active_cache)
        return score_collections

    @property
    def array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        array_collections = {}
        array_collections.update(self._validation_plan_callback.validation_plan.array_collections)
        array_collections.update(self._train_loader_array_collection_handler.active_cache)
        array_collections.update(self._val_loader_array_collection_handler.active_cache)
        return array_collections

    @property
    def plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        plot_collections = {}
        plot_collections.update(self._validation_plan_callback.validation_plan.plot_collections)
        plot_collections.update(self._train_loader_plot_collection_handler.active_cache)
        plot_collections.update(self._val_loader_plot_collection_handler.active_cache)
        return plot_collections

    @staticmethod
    @abstractmethod
    def _get_train_loader_score_callbacks() -> List[
        DataLoaderScoreCallback[ModelInputT, ModelOutputT]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_train_loader_array_callbacks() -> List[
        DataLoaderArrayCallback[ModelInputT, ModelOutputT]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_train_loader_plot_callbacks() -> List[
        DataLoaderPlotCallback[ModelInputT, ModelOutputT]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_train_loader_score_collection_callbacks() -> List[
        DataLoaderScoreCollectionCallback[ModelInputT, ModelOutputT]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_train_loader_array_collection_callbacks() -> List[
        DataLoaderArrayCollectionCallback[ModelInputT, ModelOutputT]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_train_loader_plot_collection_callbacks() -> List[
        DataLoaderPlotCollectionCallback[ModelInputT, ModelOutputT]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_val_loader_score_callbacks() -> List[
        DataLoaderScoreCallback[ModelInputT, ModelOutputT]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_val_loader_array_callbacks() -> List[
        DataLoaderArrayCallback[ModelInputT, ModelOutputT]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_val_loader_plot_callbacks() -> List[
        DataLoaderPlotCallback[ModelInputT, ModelOutputT]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_val_loader_score_collection_callbacks() -> List[
        DataLoaderScoreCollectionCallback[ModelInputT, ModelOutputT]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_val_loader_array_collection_callbacks() -> List[
        DataLoaderArrayCollectionCallback[ModelInputT, ModelOutputT]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_val_loader_plot_collection_callbacks() -> List[
        DataLoaderPlotCollectionCallback[ModelInputT, ModelOutputT]
    ]: ...

    def execute(self, model: ModelT, n_epochs_elapsed: int):
        self._execute_validation_plan_callback(
            callback=self._validation_plan_callback, model=model, n_epochs_elapsed=n_epochs_elapsed
        )
        self._execute_loader_handlers(
            ls_loader_handlers=[
                self._train_loader_score_handler,
                self._train_loader_array_handler,
                self._train_loader_plot_handler,
                self._train_loader_score_collection_handler,
                self._train_loader_array_collection_handler,
                self._train_loader_plot_collection_handler,
            ],
            model=model,
            data_loader=self._train_loader,
            n_epochs_elapsed=n_epochs_elapsed,
        )
        if self._val_loader is not None:
            self._execute_loader_handlers(
                ls_loader_handlers=[
                    self._val_loader_score_handler,
                    self._val_loader_array_handler,
                    self._val_loader_plot_handler,
                    self._val_loader_score_collection_handler,
                    self._val_loader_array_collection_handler,
                    self._val_loader_plot_collection_handler,
                ],
                model=model,
                data_loader=self._val_loader,
                n_epochs_elapsed=n_epochs_elapsed,
            )

    def clear_cache(self):
        self._validation_plan_callback.validation_plan.clear_cache()
        self._train_loader_score_handler.clear()
        self._train_loader_array_handler.clear()
        self._train_loader_plot_handler.clear()
        self._train_loader_score_collection_handler.clear()
        self._train_loader_array_collection_handler.clear()
        self._train_loader_plot_collection_handler.clear()
        self._val_loader_score_handler.clear()
        self._val_loader_array_handler.clear()
        self._val_loader_plot_handler.clear()
        self._val_loader_score_collection_handler.clear()
        self._val_loader_array_collection_handler.clear()
        self._val_loader_plot_collection_handler.clear()

    @staticmethod
    def _execute_validation_plan_callback(
        callback: ValidationPlanCallback[ModelT, ValidationPlanT, ArtifactResourcesT],
        model: ModelT,
        n_epochs_elapsed: int,
    ):
        resources = ValidationPlanCallbackResources[ModelT](step=n_epochs_elapsed, model=model)
        callback.execute(resources=resources)

    @staticmethod
    def _execute_loader_handlers(
        ls_loader_handlers: List[
            Union[
                DataLoaderScoreHandler[
                    ModelInputT,
                    ModelOutputT,
                ],
                DataLoaderArrayHandler[
                    ModelInputT,
                    ModelOutputT,
                ],
                DataLoaderPlotHandler[
                    ModelInputT,
                    ModelOutputT,
                ],
                DataLoaderScoreCollectionHandler[
                    ModelInputT,
                    ModelOutputT,
                ],
                DataLoaderArrayCollectionHandler[
                    ModelInputT,
                    ModelOutputT,
                ],
                DataLoaderPlotCollectionHandler[
                    ModelInputT,
                    ModelOutputT,
                ],
            ]
        ],
        model: Model[ModelInputT, ModelOutputT],
        data_loader: DataLoader[ModelInputT],
        n_epochs_elapsed: int,
    ):
        resources = DataLoaderCallbackResources[ModelInputT, ModelOutputT](
            step=n_epochs_elapsed, model=model, data_loader=data_loader
        )
        for loader_handler in ls_loader_handlers:
            loader_handler.execute(resources=resources)
