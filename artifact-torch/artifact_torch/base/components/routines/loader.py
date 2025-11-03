from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Mapping, Optional, Type, TypeVar

import torch
from artifact_experiment.base.entities.data_split import DataSplit
from artifact_experiment.base.tracking.client import TrackingClient
from matplotlib.figure import Figure
from numpy import ndarray
from tqdm import tqdm

from artifact_torch.base.components.callbacks.forward_hook import ForwardHookCallbackResources
from artifact_torch.base.components.plans.forward_hook import ForwardHookPlan
from artifact_torch.base.components.plans.model_io import ModelIOPlan
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.libs.utils.key_selector import KeySelector

ModelTContr = TypeVar("ModelTContr", bound=Model[Any, Any], contravariant=True)
ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
DataLoaderRoutineT = TypeVar("DataLoaderRoutineT", bound="DataLoaderRoutine")


class DataLoaderRoutine(ABC, Generic[ModelTContr, ModelInputTContr, ModelOutputTContr]):
    _verbose = True
    _progressbar_message = "Processing Data Loader"

    def __init__(
        self,
        model_io_plans: Mapping[DataSplit, ModelIOPlan[ModelInputTContr, ModelOutputTContr]],
        forward_hook_plans: Mapping[DataSplit, ForwardHookPlan[ModelTContr]],
        data_loaders: Mapping[DataSplit, DataLoader[ModelInputTContr]],
    ):
        self._model_io_plans = KeySelector.restrict_to_keys(model_io_plans, keys_from=data_loaders)
        self._forward_hook_plans = KeySelector.restrict_to_keys(
            forward_hook_plans, keys_from=data_loaders
        )
        self._data_loaders = data_loaders

    @classmethod
    def build(
        cls: Type[DataLoaderRoutineT],
        data_loaders: Mapping[DataSplit, DataLoader[ModelInputTContr]],
        tracking_client: Optional[TrackingClient] = None,
    ) -> DataLoaderRoutineT:
        model_io_plans = {
            data_split: plan
            for data_split in DataSplit
            if (
                plan := cls._get_model_io_plan(
                    data_split=data_split, tracking_client=tracking_client
                )
            )
            is not None
        }
        forward_hook_plans = {
            data_split: plan
            for data_split in DataSplit
            if (
                plan := cls._get_forward_hook_plan(
                    data_split=data_split, tracking_client=tracking_client
                )
            )
            is not None
        }
        routine = cls(
            model_io_plans=model_io_plans,
            forward_hook_plans=forward_hook_plans,
            data_loaders=data_loaders,
        )
        return routine

    @property
    def scores(self) -> Dict[str, float]:
        scores = {}
        scores.update(self._model_io_scores)
        scores.update(self._forward_hook_scores)
        return scores

    @property
    def arrays(self) -> Dict[str, ndarray]:
        arrays = {}
        arrays.update(self._model_io_arrays)
        arrays.update(self._forward_hook_arrays)
        return arrays

    @property
    def plots(self) -> Dict[str, Figure]:
        plots = {}
        plots.update(self._model_io_plots)
        plots.update(self._forward_hook_plots)
        return plots

    @property
    def score_collections(self) -> Dict[str, Dict[str, float]]:
        score_collections = {}
        score_collections.update(self._model_io_score_collections)
        score_collections.update(self._forward_hook_score_collections)
        return score_collections

    @property
    def array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        array_collections = {}
        array_collections.update(self._model_io_array_collections)
        array_collections.update(self._forward_hook_array_collections)
        return array_collections

    @property
    def plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        plot_collections = {}
        plot_collections.update(self._model_io_plot_collections)
        plot_collections.update(self._forward_hook_plot_collections)
        return plot_collections

    @property
    def _model_io_scores(self) -> Dict[str, float]:
        return {
            name: value
            for model_io_plan in self._model_io_plans.values()
            for name, value in model_io_plan.scores.items()
        }

    @property
    def _model_io_arrays(self) -> Dict[str, ndarray]:
        return {
            name: value
            for model_io_plan in self._model_io_plans.values()
            for name, value in model_io_plan.arrays.items()
        }

    @property
    def _model_io_plots(self) -> Dict[str, Figure]:
        return {
            name: value
            for model_io_plan in self._model_io_plans.values()
            for name, value in model_io_plan.plots.items()
        }

    @property
    def _model_io_score_collections(self) -> Dict[str, Dict[str, float]]:
        return {
            name: value
            for model_io_plan in self._model_io_plans.values()
            for name, value in model_io_plan.score_collections.items()
        }

    @property
    def _model_io_array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        return {
            name: value
            for model_io_plan in self._model_io_plans.values()
            for name, value in model_io_plan.array_collections.items()
        }

    @property
    def _model_io_plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        return {
            name: value
            for model_io_plan in self._model_io_plans.values()
            for name, value in model_io_plan.plot_collections.items()
        }

    @property
    def _forward_hook_scores(self) -> Dict[str, float]:
        return {
            name: value
            for forward_hook_plan in self._forward_hook_plans.values()
            for name, value in forward_hook_plan.scores.items()
        }

    @property
    def _forward_hook_arrays(self) -> Dict[str, ndarray]:
        return {
            name: value
            for forward_hook_plan in self._forward_hook_plans.values()
            for name, value in forward_hook_plan.arrays.items()
        }

    @property
    def _forward_hook_plots(self) -> Dict[str, Figure]:
        return {
            name: value
            for forward_hook_plan in self._forward_hook_plans.values()
            for name, value in forward_hook_plan.plots.items()
        }

    @property
    def _forward_hook_score_collections(self) -> Dict[str, Dict[str, float]]:
        return {
            name: value
            for forward_hook_plan in self._forward_hook_plans.values()
            for name, value in forward_hook_plan.score_collections.items()
        }

    @property
    def _forward_hook_array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        return {
            name: value
            for forward_hook_plan in self._forward_hook_plans.values()
            for name, value in forward_hook_plan.array_collections.items()
        }

    @property
    def _forward_hook_plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        return {
            name: value
            for forward_hook_plan in self._forward_hook_plans.values()
            for name, value in forward_hook_plan.plot_collections.items()
        }

    @property
    def _data_splits(self) -> List[DataSplit]:
        return list(self._data_loaders.keys())

    @staticmethod
    @abstractmethod
    def _get_model_io_plan(
        data_split: DataSplit, tracking_client: Optional[TrackingClient]
    ) -> Optional[ModelIOPlan[ModelInputTContr, ModelOutputTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_forward_hook_plan(
        data_split: DataSplit, tracking_client: Optional[TrackingClient]
    ) -> Optional[ForwardHookPlan[ModelTContr]]: ...

    def clear_cache(self):
        self._clear_model_io_cache()
        self._clear_forward_hook_cache()

    def execute(self, model: ModelTContr, n_epochs_elapsed: int):
        callback_resources = ForwardHookCallbackResources[ModelTContr](
            step=n_epochs_elapsed, model=model
        )
        for data_split in self._data_splits:
            data_loader = self._data_loaders[data_split]
            model_io_plan = self._model_io_plans.get(data_split)
            forward_hook_plan = self._forward_hook_plans.get(data_split)
            self._execute(
                model_io_plan=model_io_plan,
                forward_hook_plan=forward_hook_plan,
                callback_resources=callback_resources,
                data_loader=data_loader,
            )

    @classmethod
    def _execute(
        cls,
        model_io_plan: Optional[ModelIOPlan[ModelInputTContr, ModelOutputTContr]],
        forward_hook_plan: Optional[ForwardHookPlan[ModelTContr]],
        callback_resources: ForwardHookCallbackResources[ModelTContr],
        data_loader: DataLoader[ModelInputTContr],
    ):
        any_attached = cls._attach(
            model_io_plan=model_io_plan,
            forward_hook_plan=forward_hook_plan,
            callback_resources=callback_resources,
        )
        if any_attached:
            cls._process_data_loader(model=callback_resources.model, data_loader=data_loader)
            if model_io_plan is not None:
                model_io_plan.execute(resources=callback_resources)
            if forward_hook_plan is not None:
                forward_hook_plan.execute(resources=callback_resources)

    @staticmethod
    def _attach(
        model_io_plan: Optional[ModelIOPlan[ModelInputTContr, ModelOutputTContr]],
        forward_hook_plan: Optional[ForwardHookPlan[ModelTContr]],
        callback_resources: ForwardHookCallbackResources[ModelTContr],
    ) -> bool:
        any_attached = False
        if model_io_plan is not None:
            any_attached |= model_io_plan.attach(resources=callback_resources)
        if forward_hook_plan is not None:
            any_attached |= forward_hook_plan.attach(resources=callback_resources)
        return any_attached

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

    def _clear_model_io_cache(self):
        for plan in self._model_io_plans.values():
            plan.clear_cache()

    def _clear_forward_hook_cache(self):
        for plan in self._forward_hook_plans.values():
            plan.clear_cache()
