from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

import pandas as pd
import torch
from artifact_experiment.base.tracking.client import TrackingClient
from torch import optim

from artifact_torch.base.components.cache.cache import StandardCache
from artifact_torch.base.components.cache.score_cache import (
    ScoreCache,
)
from artifact_torch.base.components.callbacks.checkpoint import (
    CheckpointCallback,
    CheckpointCallbackResources,
)
from artifact_torch.base.components.early_stopping.stopper import EarlyStopper, StopperUpdateData
from artifact_torch.base.components.logging.progress import TrainingProgressLogger
from artifact_torch.base.components.model_tracking.tracker import (
    ModelTracker,
    ModelTrackingCriterion,
)
from artifact_torch.base.components.routines.artifact import (
    ArtifactValidationRoutine,
)
from artifact_torch.base.components.routines.batch import BatchRoutine
from artifact_torch.base.components.routines.data_loader import DataLoaderRoutine
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.base.trainer.base import TrainerBase
from artifact_torch.base.trainer.evaluation import EvaluationFlow
from artifact_torch.base.trainer.training_state import TrainingState

ModelInputT = TypeVar("ModelInputT", bound=ModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=ModelOutput)
ModelT = TypeVar("ModelT", bound=Model[Any, Any])
ModelTrackingCriterionT = TypeVar("ModelTrackingCriterionT", bound=ModelTrackingCriterion)
StopperUpdateDataT = TypeVar("StopperUpdateDataT", bound=StopperUpdateData)
CustomTrainerT = TypeVar("CustomTrainerT", bound="CustomTrainer")


class CustomTrainer(
    TrainerBase[ModelT, ModelInputT, ModelOutputT],
    Generic[
        ModelT,
        ModelInputT,
        ModelOutputT,
        ModelTrackingCriterionT,
        StopperUpdateDataT,
    ],
):
    _train_loss_key = "TRAIN_LOSS"
    _val_loss_key = "VAL_LOSS"
    _maintain_batch_scores = False

    def __init__(
        self,
        training_state: TrainingState,
        train_loader: DataLoader[ModelInputT],
        device: torch.device,
        evaluation_flow: EvaluationFlow[ModelT, ModelInputT, ModelOutputT],
        early_stopper: EarlyStopper[StopperUpdateDataT],
        model_tracker: Optional[ModelTracker[ModelTrackingCriterionT]],
        checkpoint_callback: Optional[CheckpointCallback],
        batch_routine: Optional[BatchRoutine[ModelInputT, ModelOutputT, ModelT]],
    ):
        super().__init__(training_state=training_state, train_loader=train_loader, device=device)
        self._evaluation_flow = evaluation_flow
        self._model_tracker = model_tracker
        self._early_stopper = early_stopper
        self._checkpoint_callback = checkpoint_callback
        self._batch_routine = batch_routine
        self._batch_cache = StandardCache[Any]()
        self._epoch_score_cache = ScoreCache()

    @classmethod
    def build(
        cls: Type[CustomTrainerT],
        model: ModelT,
        train_loader: DataLoader[ModelInputT],
        artifact_routine: Optional[ArtifactValidationRoutine[ModelT, Any, Any, Any, Any]] = None,
        val_loader_routine: Optional[DataLoaderRoutine[ModelInputT, ModelOutputT]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> CustomTrainerT:
        batch_routine = cls._get_batch_routine(tracking_client=tracking_client)
        train_loader_routine = cls._get_train_loader_routine(
            data_loader=train_loader, tracking_client=tracking_client
        )
        ls_loader_routines = [
            routine for routine in [train_loader_routine, val_loader_routine] if routine is not None
        ]
        trainer = cls._build(
            model=model,
            train_loader=train_loader,
            artifact_routine=artifact_routine,
            batch_routine=batch_routine,
            ls_loader_routines=ls_loader_routines,
            tracking_client=tracking_client,
        )
        return trainer

    @property
    def epoch_scores(self) -> pd.DataFrame:
        return self._epoch_score_cache.scores

    @property
    def batch_cache(self) -> StandardCache[Any]:
        return self._batch_cache

    @property
    def best_model_state(self) -> Optional[Dict[str, Any]]:
        if self._model_tracker is not None:
            return self._model_tracker.best_model_state

    @property
    def latest_model_state(self) -> Dict[str, Any]:
        return self.model.state_dict()

    @staticmethod
    @abstractmethod
    def _get_optimizer(model: ModelT) -> optim.Optimizer: ...

    @staticmethod
    @abstractmethod
    def _get_scheduler(
        optimizer: optim.Optimizer,
    ) -> Optional[optim.lr_scheduler._LRScheduler]: ...

    @staticmethod
    @abstractmethod
    def _get_device() -> torch.device: ...

    @staticmethod
    @abstractmethod
    def _get_model_tracker() -> Optional[ModelTracker[ModelTrackingCriterionT]]: ...

    @abstractmethod
    def _get_model_tracking_criterion(self) -> Optional[ModelTrackingCriterionT]: ...

    @staticmethod
    @abstractmethod
    def _get_early_stopper() -> EarlyStopper[StopperUpdateDataT]: ...

    @abstractmethod
    def _get_stopper_update_data(self) -> StopperUpdateDataT: ...

    @staticmethod
    @abstractmethod
    def _get_checkpoint_callback(
        tracking_client: Optional[TrackingClient],
    ) -> Optional[CheckpointCallback]: ...

    @staticmethod
    @abstractmethod
    def _get_batch_routine(
        tracking_client: Optional[TrackingClient],
    ) -> Optional[BatchRoutine[ModelInputT, ModelOutputT, ModelT]]: ...

    @staticmethod
    @abstractmethod
    def _get_train_loader_routine(
        data_loader: DataLoader[ModelInputT],
        tracking_client: Optional[TrackingClient],
    ) -> Optional[
        DataLoaderRoutine[
            ModelInputT,
            ModelOutputT,
        ]
    ]: ...

    def _should_stop(self) -> bool:
        return self._early_stopper.stopped

    def _execute_batch_postprocessing(
        self,
        model_input: ModelInputT,
        model_output: ModelOutputT,
        batch_idx: int,
    ):
        self._execute_batch_routine(
            model_input=model_input,
            model_output=model_output,
            batch_idx=batch_idx,
        )

    def _execute_epoch_postprocessing(self):
        super()._execute_epoch_postprocessing()
        self._execute_evaluation_flow()
        self._execute_checkpoint_callback()
        self._update_tracker()
        self._update_stopper()

    def _execute_batch_routine(
        self,
        model_input: ModelInputT,
        model_output: ModelOutputT,
        batch_idx: int,
    ):
        if self._batch_routine is not None:
            self._batch_routine.execute(
                model_input=model_input,
                model_output=model_output,
                model=self.model,
                batch_idx=batch_idx,
            )
            self._batch_cache.append(items=self._batch_routine.cache)

    def _execute_evaluation_flow(self):
        self._evaluation_flow.clear_cache()
        self._evaluation_flow.execute(model=self.model, n_epochs_elapsed=self.n_epochs_elapsed)
        self._epoch_score_cache.append(self._evaluation_flow.scores)

    def _execute_checkpoint_callback(self):
        if self._checkpoint_callback is not None:
            checkpoint = self._get_checkpoint()
            checkpoint_callback_resources = CheckpointCallbackResources(
                step=self._n_epochs_elapsed, checkpoint=checkpoint
            )
            self._checkpoint_callback.execute(resources=checkpoint_callback_resources)

    def _update_tracker(self):
        if self._model_tracker is not None:
            criterion = self._get_model_tracking_criterion()
            assert criterion is not None
            self._model_tracker.update(
                model=self.model,
                criterion=criterion,
                epoch=self.n_epochs_elapsed,
            )

    def _update_stopper(self):
        update_data = self._get_stopper_update_data()
        self._early_stopper.update(update_data=update_data)

    def _get_progress_bar_description(self) -> str:
        train_loss = self._epoch_score_cache.get_latest_non_null(key=self._train_loss_key)
        val_loss = self._epoch_score_cache.get_latest_non_null(key=self._val_loss_key)
        desc = TrainingProgressLogger.get_progress_update(
            n_epochs_elapsed=self.n_epochs_elapsed, train_loss=train_loss, val_loss=val_loss
        )
        return desc

    @classmethod
    def _build(
        cls: Type[CustomTrainerT],
        model: ModelT,
        train_loader: DataLoader[ModelInputT],
        batch_routine: Optional[BatchRoutine[ModelInputT, ModelOutputT, ModelT]] = None,
        artifact_routine: Optional[ArtifactValidationRoutine[ModelT, Any, Any, Any, Any]] = None,
        ls_loader_routines: Optional[List[DataLoaderRoutine[ModelInputT, ModelOutputT]]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> CustomTrainerT:
        optimizer = cls._get_optimizer(model=model)
        scheduler = cls._get_scheduler(optimizer=optimizer)
        training_state = TrainingState[ModelT](
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        device = cls._get_device()
        evaluation_flow = EvaluationFlow(
            artifact_routine=artifact_routine,
            ls_loader_routines=ls_loader_routines,
        )
        early_stopper = cls._get_early_stopper()
        model_tracker = cls._get_model_tracker()
        checkpoint_callback = cls._get_checkpoint_callback(tracking_client=tracking_client)
        trainer = cls(
            training_state=training_state,
            train_loader=train_loader,
            device=device,
            evaluation_flow=evaluation_flow,
            model_tracker=model_tracker,
            early_stopper=early_stopper,
            checkpoint_callback=checkpoint_callback,
            batch_routine=batch_routine,
        )
        return trainer
