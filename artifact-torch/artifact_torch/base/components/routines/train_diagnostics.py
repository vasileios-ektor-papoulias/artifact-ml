from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Type, TypeVar

from artifact_experiment.base.tracking.client import TrackingClient
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_torch.base.components.callbacks.hook import HookCallbackResources
from artifact_torch.base.components.plans.backward_hook import BackwardHookPlan
from artifact_torch.base.components.plans.forward_hook import ForwardHookPlan
from artifact_torch.base.components.plans.model_io import ModelIOPlan
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelTContr = TypeVar("ModelTContr", bound=Model[Any, Any], contravariant=True)
ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
TrainDiagnosticsRoutineT = TypeVar(
    "TrainDiagnosticsRoutineT", bound="TrainDiagnosticsRoutine[Any, Any, Any]"
)


class TrainDiagnosticsRoutine(ABC, Generic[ModelTContr, ModelInputTContr, ModelOutputTContr]):
    _callback_trigger_identifier = "EPOCH"

    def __init__(
        self,
        model_io_plan: Optional[ModelIOPlan[ModelInputTContr, ModelOutputTContr]],
        forward_hook_plan: Optional[ForwardHookPlan[ModelTContr]],
        backward_hook_plan: Optional[BackwardHookPlan[ModelTContr]],
    ):
        self._model_io_plan = model_io_plan
        self._forward_hook_plan = forward_hook_plan
        self._backward_hook_plan = backward_hook_plan

    @classmethod
    def build(
        cls: Type[TrainDiagnosticsRoutineT], tracking_client: Optional[TrackingClient] = None
    ) -> TrainDiagnosticsRoutineT:
        model_io_plan = cls._get_model_io_plan(tracking_client=tracking_client)
        forward_hook_plan = cls._get_forward_hook_plan(tracking_client=tracking_client)
        backward_hook_plan = cls._get_backward_hook_plan(tracking_client=tracking_client)
        routine = cls(
            model_io_plan=model_io_plan,
            forward_hook_plan=forward_hook_plan,
            backward_hook_plan=backward_hook_plan,
        )
        return routine

    @property
    def scores(self) -> Dict[str, float]:
        scores = {}
        scores.update(self._model_io_scores)
        scores.update(self._forward_hook_scores)
        scores.update(self._backward_hook_scores)
        return scores

    @property
    def arrays(self) -> Dict[str, ndarray]:
        arrays = {}
        arrays.update(self._model_io_arrays)
        arrays.update(self._forward_hook_arrays)
        arrays.update(self._backward_hook_arrays)
        return arrays

    @property
    def plots(self) -> Dict[str, Figure]:
        plots = {}
        plots.update(self._model_io_plots)
        plots.update(self._forward_hook_plots)
        plots.update(self._backward_hook_plots)
        return plots

    @property
    def score_collections(self) -> Dict[str, Dict[str, float]]:
        score_collections = {}
        score_collections.update(self._model_io_score_collections)
        score_collections.update(self._forward_hook_score_collections)
        score_collections.update(self._backward_hook_score_collections)
        return score_collections

    @property
    def array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        array_collections = {}
        array_collections.update(self._model_io_array_collections)
        array_collections.update(self._forward_hook_array_collections)
        array_collections.update(self._backward_hook_array_collections)
        return array_collections

    @property
    def plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        plot_collections = {}
        plot_collections.update(self._model_io_plot_collections)
        plot_collections.update(self._forward_hook_plot_collections)
        plot_collections.update(self._backward_hook_plot_collections)
        return plot_collections

    @property
    def _model_io_scores(self) -> Dict[str, float]:
        return self._model_io_plan.scores if self._model_io_plan is not None else {}

    @property
    def _model_io_arrays(self) -> Dict[str, ndarray]:
        return self._model_io_plan.arrays if self._model_io_plan is not None else {}

    @property
    def _model_io_plots(self) -> Dict[str, Figure]:
        return self._model_io_plan.plots if self._model_io_plan is not None else {}

    @property
    def _model_io_score_collections(self) -> Dict[str, Dict[str, float]]:
        return self._model_io_plan.score_collections if self._model_io_plan is not None else {}

    @property
    def _model_io_array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        return self._model_io_plan.array_collections if self._model_io_plan is not None else {}

    @property
    def _model_io_plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        return self._model_io_plan.plot_collections if self._model_io_plan is not None else {}

    @property
    def _forward_hook_scores(self) -> Dict[str, float]:
        return self._forward_hook_plan.scores if self._forward_hook_plan is not None else {}

    @property
    def _forward_hook_arrays(self) -> Dict[str, ndarray]:
        return self._forward_hook_plan.arrays if self._forward_hook_plan is not None else {}

    @property
    def _forward_hook_plots(self) -> Dict[str, Figure]:
        return self._forward_hook_plan.plots if self._forward_hook_plan is not None else {}

    @property
    def _forward_hook_score_collections(self) -> Dict[str, Dict[str, float]]:
        return (
            self._forward_hook_plan.score_collections if self._forward_hook_plan is not None else {}
        )

    @property
    def _forward_hook_array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        return (
            self._forward_hook_plan.array_collections if self._forward_hook_plan is not None else {}
        )

    @property
    def _forward_hook_plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        return (
            self._forward_hook_plan.plot_collections if self._forward_hook_plan is not None else {}
        )

    @property
    def _backward_hook_scores(self) -> Dict[str, float]:
        return self._backward_hook_plan.scores if self._backward_hook_plan is not None else {}

    @property
    def _backward_hook_arrays(self) -> Dict[str, ndarray]:
        return self._backward_hook_plan.arrays if self._backward_hook_plan is not None else {}

    @property
    def _backward_hook_plots(self) -> Dict[str, Figure]:
        return self._backward_hook_plan.plots if self._backward_hook_plan is not None else {}

    @property
    def _backward_hook_score_collections(self) -> Dict[str, Dict[str, float]]:
        return (
            self._backward_hook_plan.score_collections
            if self._backward_hook_plan is not None
            else {}
        )

    @property
    def _backward_hook_array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        return (
            self._backward_hook_plan.array_collections
            if self._backward_hook_plan is not None
            else {}
        )

    @property
    def _backward_hook_plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        return (
            self._backward_hook_plan.plot_collections
            if self._backward_hook_plan is not None
            else {}
        )

    @staticmethod
    @abstractmethod
    def _get_model_io_plan(
        tracking_client: Optional[TrackingClient],
    ) -> Optional[ModelIOPlan[ModelInputTContr, ModelOutputTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_forward_hook_plan(
        tracking_client: Optional[TrackingClient],
    ) -> Optional[ForwardHookPlan[ModelTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_backward_hook_plan(
        tracking_client: Optional[TrackingClient],
    ) -> Optional[BackwardHookPlan[ModelTContr]]: ...

    def clear_cache(self):
        self._clear_model_io_cache()
        self._clear_forward_hook_cache()
        self._clear_backward_hook_cache()

    def execute(self, model: ModelTContr, n_epochs_elapsed: int):
        resources = HookCallbackResources[ModelTContr](
            model=model, step=n_epochs_elapsed, trigger=self._callback_trigger_identifier
        )
        if self._model_io_plan is not None:
            self._model_io_plan.execute(resources=resources)
        if self._forward_hook_plan is not None:
            self._forward_hook_plan.execute(resources=resources)
        if self._backward_hook_plan is not None:
            self._backward_hook_plan.execute(resources=resources)

    def attach(self, model: ModelTContr, n_epochs_elapsed: int) -> bool:
        resources = HookCallbackResources[ModelTContr](model=model, step=n_epochs_elapsed)
        any_attached = False
        if self._model_io_plan is not None:
            any_attached |= self._model_io_plan.attach(resources=resources)
        if self._forward_hook_plan is not None:
            any_attached |= self._forward_hook_plan.attach(resources=resources)
        if self._backward_hook_plan is not None:
            any_attached |= self._backward_hook_plan.attach(resources=resources)
        return any_attached

    def _clear_model_io_cache(self):
        if self._model_io_plan is not None:
            self._model_io_plan.clear_cache()

    def _clear_forward_hook_cache(self):
        if self._forward_hook_plan is not None:
            self._forward_hook_plan.clear_cache()

    def _clear_backward_hook_cache(self):
        if self._backward_hook_plan is not None:
            self._backward_hook_plan.clear_cache()
