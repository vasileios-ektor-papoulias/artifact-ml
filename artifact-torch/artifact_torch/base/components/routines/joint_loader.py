from typing import Any, Dict, Generic, Optional, TypeVar

import torch
from matplotlib.figure import Figure
from numpy import ndarray
from tqdm import tqdm

from artifact_torch.base.components.routines.loader import DataLoaderRoutine
from artifact_torch.base.components.routines.loader_hook import DataLoaderHookRoutine
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelTContr = TypeVar("ModelTContr", bound=Model[Any, Any], contravariant=True)
ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)


class JointDataLoaderRoutine(Generic[ModelTContr, ModelInputTContr, ModelOutputTContr]):
    _verbose = True
    _progressbar_message = "Processing Data Loader (Joint)"

    def __init__(
        self,
        data_loader: DataLoader[ModelInputTContr],
        loader_routine: Optional[DataLoaderRoutine[ModelInputTContr, ModelOutputTContr]] = None,
        loader_hook_routine: Optional[
            DataLoaderHookRoutine[ModelTContr, ModelInputTContr, ModelOutputTContr]
        ] = None,
    ):
        self._data_loader = data_loader
        self._loader_routine = loader_routine
        self._loader_hook_routine = loader_hook_routine

    @property
    def scores(self) -> Dict[str, float]:
        scores = {}
        if self._loader_routine is not None:
            scores.update(self._loader_routine.scores)
        if self._loader_hook_routine is not None:
            scores.update(self._loader_hook_routine.scores)
        return scores

    @property
    def arrays(self) -> Dict[str, ndarray]:
        arrays = {}
        if self._loader_routine is not None:
            arrays.update(self._loader_routine.arrays)
        if self._loader_hook_routine is not None:
            arrays.update(self._loader_hook_routine.arrays)
        return arrays

    @property
    def plots(self) -> Dict[str, Figure]:
        plots = {}
        if self._loader_routine is not None:
            plots.update(self._loader_routine.plots)
        if self._loader_hook_routine is not None:
            plots.update(self._loader_hook_routine.plots)
        return plots

    @property
    def score_collections(self) -> Dict[str, Dict[str, float]]:
        score_collections = {}
        if self._loader_routine is not None:
            score_collections.update(self._loader_routine.score_collections)
        if self._loader_hook_routine is not None:
            score_collections.update(self._loader_hook_routine.score_collections)
        return score_collections

    @property
    def array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        array_collections = {}
        if self._loader_routine is not None:
            array_collections.update(self._loader_routine.array_collections)
        if self._loader_hook_routine is not None:
            array_collections.update(self._loader_hook_routine.array_collections)
        return array_collections

    @property
    def plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        plot_collections = {}
        if self._loader_routine is not None:
            plot_collections.update(self._loader_routine.plot_collections)
        if self._loader_hook_routine is not None:
            plot_collections.update(self._loader_hook_routine.plot_collections)
        return plot_collections

    @property
    def _n_callbacks(self) -> int:
        n_callbacks = 0
        if self._loader_routine is not None:
            n_callbacks += self._loader_routine.n_callbacks
        if self._loader_hook_routine is not None:
            n_callbacks += self._loader_hook_routine.n_callbacks
        return n_callbacks

    @property
    def _has_callbacks(self) -> bool:
        return self._n_callbacks != 0

    def clear_cache(self):
        self._clear_cache()

    def execute(
        self,
        model: ModelTContr,
        n_epochs_elapsed: int,
    ):
        if self._has_callbacks:
            self._execute_parallel(
                model=model, data_loader=self._data_loader, n_epochs_elapsed=n_epochs_elapsed
            )

    def _execute_sequential(self, model: ModelTContr, n_epochs_elapsed: int):
        if self._loader_routine is not None:
            self._loader_routine.execute(
                model=model, data_loader=self._data_loader, n_epochs_elapsed=n_epochs_elapsed
            )
        if self._loader_hook_routine is not None:
            self._loader_hook_routine.execute(
                model=model, data_loader=self._data_loader, n_epochs_elapsed=n_epochs_elapsed
            )

    def _execute_parallel(
        self, model: ModelTContr, data_loader: DataLoader[ModelInputTContr], n_epochs_elapsed: int
    ):
        self._clear_cache()
        if self._should_trigger(n_epochs_elapsed=n_epochs_elapsed):
            self._attach_hooks(model=model, n_epochs_elapsed=n_epochs_elapsed)
            model.eval()
            try:
                with torch.no_grad():
                    for model_input in tqdm(
                        data_loader,
                        desc=self._progressbar_message,
                        disable=not self._verbose,
                        leave=False,
                    ):
                        model_output = model(model_input)
                        self._process_batch(
                            model_input=model_input,
                            model_output=model_output,
                            n_epochs_elapsed=n_epochs_elapsed,
                        )
            finally:
                self._detach_hooks(n_epochs_elapsed=n_epochs_elapsed)
            self._finalize(n_epochs_elapsed=n_epochs_elapsed)
            self._export()

    def _clear_cache(self):
        if self._loader_routine is not None:
            self._loader_routine.clear_cache()
        if self._loader_hook_routine is not None:
            self._loader_hook_routine.clear_cache()

    def _should_trigger(self, n_epochs_elapsed: int) -> bool:
        should_trigger = (
            self._loader_routine is not None
            and self._loader_routine.should_trigger(n_epochs_elapsed=n_epochs_elapsed)
        ) and (
            self._loader_hook_routine is not None
            and self._loader_hook_routine.should_trigger(n_epochs_elapsed=n_epochs_elapsed)
        )
        return should_trigger

    def _attach_hooks(self, model: ModelTContr, n_epochs_elapsed: int):
        if self._loader_hook_routine is not None:
            self._loader_hook_routine.attach_hooks(model=model, n_epochs_elapsed=n_epochs_elapsed)

    def _process_batch(
        self, model_input: ModelInputTContr, model_output: ModelOutputTContr, n_epochs_elapsed: int
    ):
        if self._loader_routine is not None:
            self._loader_routine.process_batch(
                model_input=model_input,
                model_output=model_output,
                n_epochs_elapsed=n_epochs_elapsed,
            )

    def _detach_hooks(self, n_epochs_elapsed: int):
        if self._loader_hook_routine is not None:
            self._loader_hook_routine.detach_hooks(n_epochs_elapsed=n_epochs_elapsed)

    def _finalize(self, n_epochs_elapsed: int):
        if self._loader_routine is not None:
            self._loader_routine.finalize(n_epochs_elapsed=n_epochs_elapsed)
        if self._loader_hook_routine is not None:
            self._loader_hook_routine.finalize(n_epochs_elapsed=n_epochs_elapsed)

    def _export(self):
        if self._loader_routine is not None:
            self._loader_routine.export()
        if self._loader_hook_routine is not None:
            self._loader_hook_routine.export()
