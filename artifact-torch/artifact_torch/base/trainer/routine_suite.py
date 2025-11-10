from typing import Any, Dict, Generic, Optional, TypeVar

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_torch.base.components.routines.artifact import ArtifactRoutine
from artifact_torch.base.components.routines.base import RoutineResources
from artifact_torch.base.components.routines.loader import DataLoaderRoutine
from artifact_torch.base.components.routines.train_diagnostics import TrainDiagnosticsRoutine
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelTContr = TypeVar("ModelTContr", bound=Model[Any, Any], contravariant=True)
ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)


class RoutineSuite(Generic[ModelTContr, ModelInputTContr, ModelOutputTContr]):
    def __init__(
        self,
        train_diagnostics_routine: Optional[
            TrainDiagnosticsRoutine[ModelTContr, ModelInputTContr, ModelOutputTContr]
        ] = None,
        loader_routine: Optional[
            DataLoaderRoutine[ModelTContr, ModelInputTContr, ModelOutputTContr]
        ] = None,
        artifact_routine: Optional[ArtifactRoutine[ModelTContr, Any, Any, Any, Any, Any]] = None,
    ):
        self._train_diagnostics_routine = train_diagnostics_routine
        self._loader_routine = loader_routine
        self._artifact_routine = artifact_routine

    @property
    def scores(self) -> Dict[str, float]:
        scores = {}
        if self._train_diagnostics_routine is not None:
            scores.update(self._train_diagnostics_routine.scores)
        if self._loader_routine is not None:
            scores.update(self._loader_routine.scores)
        if self._artifact_routine is not None:
            scores.update(self._artifact_routine.scores)
        return scores

    @property
    def arrays(self) -> Dict[str, ndarray]:
        arrays = {}
        if self._train_diagnostics_routine is not None:
            arrays.update(self._train_diagnostics_routine.arrays)
        if self._loader_routine is not None:
            arrays.update(self._loader_routine.arrays)
        if self._artifact_routine is not None:
            arrays.update(self._artifact_routine.arrays)
        return arrays

    @property
    def plots(self) -> Dict[str, Figure]:
        plots = {}
        if self._train_diagnostics_routine is not None:
            plots.update(self._train_diagnostics_routine.plots)
        if self._loader_routine is not None:
            plots.update(self._loader_routine.plots)
        if self._artifact_routine is not None:
            plots.update(self._artifact_routine.plots)
        return plots

    @property
    def score_collections(self) -> Dict[str, Dict[str, float]]:
        score_collections = {}
        if self._train_diagnostics_routine is not None:
            score_collections.update(self._train_diagnostics_routine.score_collections)
        if self._loader_routine is not None:
            score_collections.update(self._loader_routine.score_collections)
        if self._artifact_routine is not None:
            score_collections.update(self._artifact_routine.score_collections)
        return score_collections

    @property
    def array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        array_collections = {}
        if self._train_diagnostics_routine is not None:
            array_collections.update(self._train_diagnostics_routine.array_collections)
        if self._loader_routine is not None:
            array_collections.update(self._loader_routine.array_collections)
        if self._artifact_routine is not None:
            array_collections.update(self._artifact_routine.array_collections)
        return array_collections

    @property
    def plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        plot_collections = {}
        if self._train_diagnostics_routine is not None:
            plot_collections.update(self._train_diagnostics_routine.plot_collections)
        if self._loader_routine is not None:
            plot_collections.update(self._loader_routine.plot_collections)
        if self._artifact_routine is not None:
            plot_collections.update(self._artifact_routine.plot_collections)
        return plot_collections

    def attach_train_diagnostics_hooks(self, model: ModelTContr, n_epochs_elapsed: int):
        if self._train_diagnostics_routine is not None:
            self._train_diagnostics_routine.attach(model=model, n_epochs_elapsed=n_epochs_elapsed)

    def execute(self, model: ModelTContr, n_epochs_elapsed: int):
        resources = RoutineResources(model=model, n_epochs_elapsed=n_epochs_elapsed)
        if self._train_diagnostics_routine is not None:
            self._train_diagnostics_routine.execute(resources=resources)
        if self._loader_routine is not None:
            self._loader_routine.execute(resources=resources)
        if self._artifact_routine is not None:
            self._artifact_routine.execute(resources=resources)

    def clear_cache(self):
        if self._train_diagnostics_routine is not None:
            self._train_diagnostics_routine.clear_cache()
        if self._loader_routine is not None:
            self._loader_routine.clear_cache()
        if self._artifact_routine is not None:
            self._artifact_routine.clear_cache()
