from typing import Any, Dict, Generic, Optional, Sequence, TypeVar

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_torch.base.components.routines.artifact import ArtifactRoutine
from artifact_torch.base.components.routines.loader import DataLoaderRoutine
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelTContr = TypeVar("ModelTContr", bound=Model, contravariant=True)
ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)


class EpochEndFlow(Generic[ModelTContr, ModelInputTContr, ModelOutputTContr]):
    def __init__(
        self,
        artifact_routine: Optional[ArtifactRoutine[ModelTContr, Any, Any, Any, Any]] = None,
        loader_routines: Optional[
            Sequence[DataLoaderRoutine[ModelInputTContr, ModelOutputTContr]]
        ] = None,
    ):
        self._artifact_routine = artifact_routine
        if loader_routines is None:
            loader_routines = []
        self._loader_routines = loader_routines

    @property
    def scores(self) -> Dict[str, float]:
        scores = {}
        if self._artifact_routine is not None:
            scores.update(self._artifact_routine.scores)
        for loader_routine in self._loader_routines:
            scores.update(loader_routine.scores)
        return scores

    @property
    def arrays(self) -> Dict[str, ndarray]:
        arrays = {}
        if self._artifact_routine is not None:
            arrays.update(self._artifact_routine.arrays)
        for loader_routine in self._loader_routines:
            arrays.update(loader_routine.arrays)
        return arrays

    @property
    def plots(self) -> Dict[str, Figure]:
        plots = {}
        if self._artifact_routine is not None:
            plots.update(self._artifact_routine.plots)
        for loader_routine in self._loader_routines:
            plots.update(loader_routine.plots)
        return plots

    @property
    def score_collections(self) -> Dict[str, Dict[str, float]]:
        score_collections = {}
        if self._artifact_routine is not None:
            score_collections.update(self._artifact_routine.score_collections)
        for loader_routine in self._loader_routines:
            score_collections.update(loader_routine.score_collections)
        return score_collections

    @property
    def array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        array_collections = {}
        if self._artifact_routine is not None:
            array_collections.update(self._artifact_routine.array_collections)
        for loader_routine in self._loader_routines:
            array_collections.update(loader_routine.array_collections)
        return array_collections

    @property
    def plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        plot_collections = {}
        if self._artifact_routine is not None:
            plot_collections.update(self._artifact_routine.plot_collections)
        for loader_routine in self._loader_routines:
            plot_collections.update(loader_routine.plot_collections)
        return plot_collections

    def execute(self, model: ModelTContr, n_epochs_elapsed: int):
        if self._artifact_routine is not None:
            self._artifact_routine.execute(model=model, n_epochs_elapsed=n_epochs_elapsed)
        for loader_routine in self._loader_routines:
            loader_routine.execute(model=model, n_epochs_elapsed=n_epochs_elapsed)

    def clear_cache(self):
        if self._artifact_routine is not None:
            self._artifact_routine.clear_cache()
        for loader_routine in self._loader_routines:
            loader_routine.clear_cache()
