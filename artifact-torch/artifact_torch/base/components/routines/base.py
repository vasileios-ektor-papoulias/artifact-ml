from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Mapping, Sequence, TypeVar

from artifact_experiment.base.components.plans.base import CallbackExecutionPlan

from artifact_torch.base.model.base import Model

ModelTCov = TypeVar("ModelTCov", bound=Model[Any, Any], covariant=True)


@dataclass(frozen=True)
class RoutineResources(Generic[ModelTCov]):
    model: ModelTCov
    n_epochs_elapsed: int


ModelTContr = TypeVar("ModelTContr", bound=Model[Any, Any], contravariant=True)


class PlanExecutionRoutine(ABC, Generic[ModelTContr]):
    def __init__(self, plans: Sequence[CallbackExecutionPlan]):
        self._plans = plans

    @property
    def scores(self) -> Mapping[str, float]:
        scores = {}
        for plan in self._plans:
            scores.update(plan.scores)
        return scores

    @property
    def arrays(self) -> Mapping[str, Array]:
        arrays = {}
        for plan in self._plans:
            arrays.update(plan.arrays)
        return arrays

    @property
    def plots(self) -> Mapping[str, Figure]:
        plots = {}
        for plan in self._plans:
            plots.update(plan.plots)
        return plots

    @property
    def score_collections(self) -> Mapping[str, Dict[str, float]]:
        score_collections = {}
        for plan in self._plans:
            score_collections.update(plan.score_collections)
        return score_collections

    @property
    def array_collections(self) -> Mapping[str, Dict[str, Array]]:
        array_collections = {}
        for plan in self._plans:
            array_collections.update(plan.array_collections)
        return array_collections

    @property
    def plot_collections(self) -> Mapping[str, Dict[str, Figure]]:
        plot_collections = {}
        for plan in self._plans:
            plot_collections.update(plan.plot_collections)
        return plot_collections

    @abstractmethod
    def execute(self, resources: RoutineResources[ModelTContr]): ...

    def clear_cache(self):
        for plan in self._plans:
            plan.clear_cache()
