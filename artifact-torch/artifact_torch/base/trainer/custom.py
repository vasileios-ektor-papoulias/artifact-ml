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
from artifact_torch.base.components.callbacks.batch import (
    BatchCallback,
    BatchCallbackHandler,
    BatchCallbackResources,
)
from artifact_torch.base.components.callbacks.checkpoint import (
    CheckpointCallback,
    CheckpointCallbackResources,
)
from artifact_torch.base.components.early_stopping.stopper import EarlyStopper, StopperUpdateData
from artifact_torch.base.components.logging.trainer_logger import TrainerLogger
from artifact_torch.base.components.model_tracking.tracker import (
    ModelTracker,
    ModelTrackingCriterion,
)
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.base.trainer.base import TrainerBase
from artifact_torch.base.trainer.training_state import TrainingState
from artifact_torch.base.trainer.validation_routine import ValidationRoutine

ModelInputT = TypeVar("ModelInputT", bound=ModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=ModelOutput)
ModelT = TypeVar("ModelT", bound=Model[Any, Any])
ValidationRoutineT = TypeVar("ValidationRoutineT", bound=ValidationRoutine)
ModelTrackingCriterionT = TypeVar("ModelTrackingCriterionT", bound=ModelTrackingCriterion)
StopperUpdateDataT = TypeVar("StopperUpdateDataT", bound=StopperUpdateData)
CustomTrainerT = TypeVar("CustomTrainerT", bound="CustomTrainer")


class CustomTrainer(
    TrainerBase[ModelT, ModelInputT, ModelOutputT],
    Generic[
        ModelT,
        ModelInputT,
        ModelOutputT,
        ValidationRoutineT,
        ModelTrackingCriterionT,
        StopperUpdateDataT,
    ],
):
    _maintain_batch_scores = False

    def __init__(
        self,
        training_state: TrainingState,
        train_loader: DataLoader[ModelInputT],
        device: torch.device,
        validation_routine: ValidationRoutineT,
        model_tracker: Optional[ModelTracker[ModelTrackingCriterionT]],
        early_stopper: EarlyStopper[StopperUpdateDataT],
        checkpoint_callback: Optional[CheckpointCallback],
        batch_callback_handler: BatchCallbackHandler[ModelInputT, ModelOutputT, Any],
    ):
        super().__init__(training_state=training_state, train_loader=train_loader, device=device)
        self._validation_routine = validation_routine
        self._model_tracker = model_tracker
        self._early_stopper = early_stopper
        self._checkpoint_callback = checkpoint_callback
        self._batch_callback_handler = batch_callback_handler
        self._batch_cache = StandardCache[Any]()
        self._epoch_score_cache = ScoreCache()

    @classmethod
    def build(
        cls: Type[CustomTrainerT],
        model: ModelT,
        train_loader: DataLoader[ModelInputT],
        validation_routine: ValidationRoutineT,
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
        early_stopper = cls._get_early_stopper()
        model_tracker = cls._get_model_tracker()

        checkpoint_callback = cls._get_checkpoint_callback(tracking_client=tracking_client)
        ls_batch_callbacks = cls._get_batch_callbacks(tracking_client=tracking_client)
        batch_callback_handler = BatchCallbackHandler[ModelInputT, ModelOutputT, Any](
            ls_callbacks=ls_batch_callbacks
        )
        trainer = cls(
            training_state=training_state,
            train_loader=train_loader,
            device=device,
            validation_routine=validation_routine,
            model_tracker=model_tracker,
            early_stopper=early_stopper,
            checkpoint_callback=checkpoint_callback,
            batch_callback_handler=batch_callback_handler,
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
    def _get_batch_callbacks(
        tracking_client: Optional[TrackingClient],
    ) -> List[BatchCallback[ModelInputT, ModelOutputT, Any]]: ...

    @abstractmethod
    def _get_progress_bar_description(self) -> str:
        desc = TrainerLogger.get_progress_bar_desc(n_epochs_elapsed=self.n_epochs_elapsed)
        return desc

    def _should_stop(self) -> bool:
        return self._early_stopper.stopped

    def _execute_batch_postprocessing(
        self, batch_idx: int, batch: ModelInputT, model_output: ModelOutputT
    ):
        self._execute_batch_callbacks(
            batch_idx=batch_idx,
            batch=batch,
            model_output=model_output,
        )

    def _execute_epoch_postprocessing(self):
        self._execute_validation_routine()
        self._execute_checkpoint_callback()
        self._update_tracker()
        self._update_stopper()
        super()._execute_epoch_postprocessing()

    def _execute_batch_callbacks(
        self,
        batch_idx: int,
        batch: ModelInputT,
        model_output: ModelOutputT,
    ):
        batch_callback_resources = BatchCallbackResources[ModelInputT, ModelOutputT](
            step=batch_idx, model_input=batch, model_output=model_output
        )
        self._batch_callback_handler.execute(resources=batch_callback_resources)
        self._batch_cache.append(items=self._batch_callback_handler.cache)

    def _execute_validation_routine(self):
        self._validation_routine.execute(model=self.model, n_epochs_elapsed=self.n_epochs_elapsed)

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
