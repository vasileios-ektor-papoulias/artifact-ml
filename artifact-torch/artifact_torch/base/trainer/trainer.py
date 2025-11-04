from abc import abstractmethod
from typing import Any, Dict, Generic, Optional, Type, TypeVar

import pandas as pd
import torch
from artifact_experiment.base.tracking.client import TrackingClient
from torch import optim

from artifact_torch.base.components.cache.score_cache import ScoreCache
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
from artifact_torch.base.components.routines.artifact import ArtifactRoutine
from artifact_torch.base.components.routines.loader import DataLoaderRoutine
from artifact_torch.base.components.routines.train_diagnostics import TrainDiagnosticsRoutine
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.base.trainer.base import TrainerBase
from artifact_torch.base.trainer.routine_suite import RoutineSuite
from artifact_torch.base.trainer.training_state import TrainingState

ModelTContr = TypeVar("ModelTContr", bound=Model[Any, Any], contravariant=True)
ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
StopperUpdateDataT = TypeVar("StopperUpdateDataT", bound=StopperUpdateData)
ModelTrackingCriterionT = TypeVar("ModelTrackingCriterionT", bound=ModelTrackingCriterion)
TrainerT = TypeVar("TrainerT", bound="Trainer")


class Trainer(
    TrainerBase[ModelTContr, ModelInputTContr, ModelOutputTContr],
    Generic[
        ModelTContr,
        ModelInputTContr,
        ModelOutputTContr,
        StopperUpdateDataT,
        ModelTrackingCriterionT,
    ],
):
    _train_loss_key = "loss_train"
    _val_loss_key = "loss_val"

    def __init__(
        self,
        training_state: TrainingState[ModelTContr],
        train_loader: DataLoader[ModelInputTContr],
        device: torch.device,
        early_stopper: EarlyStopper[StopperUpdateDataT],
        model_tracker: Optional[ModelTracker[ModelTrackingCriterionT]],
        checkpoint_callback: Optional[CheckpointCallback],
        routine_suite: RoutineSuite[ModelTContr, ModelInputTContr, ModelOutputTContr],
    ):
        super().__init__(training_state=training_state, train_loader=train_loader, device=device)
        self._early_stopper = early_stopper
        self._model_tracker = model_tracker
        self._checkpoint_callback = checkpoint_callback
        self._routine_suite = routine_suite
        self._epoch_score_cache = ScoreCache()

    @classmethod
    def build(
        cls: Type[TrainerT],
        model: ModelTContr,
        train_loader: DataLoader[ModelInputTContr],
        train_diagnostics_routine: Optional[
            TrainDiagnosticsRoutine[ModelTContr, ModelInputTContr, ModelOutputTContr]
        ] = None,
        loader_routine: Optional[
            DataLoaderRoutine[ModelTContr, ModelInputTContr, ModelOutputTContr]
        ] = None,
        artifact_routine: Optional[ArtifactRoutine[ModelTContr, Any, Any, Any, Any]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> TrainerT:
        optimizer = cls._get_optimizer(model=model)
        scheduler = cls._get_scheduler(optimizer=optimizer)
        training_state = TrainingState[ModelTContr](
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        device = cls._get_device()
        early_stopper = cls._get_early_stopper()
        model_tracker = cls._get_model_tracker()
        checkpoint_callback = cls._get_checkpoint_callback(tracking_client=tracking_client)
        routine_suite = RoutineSuite(
            train_diagnostics_routine=train_diagnostics_routine,
            artifact_routine=artifact_routine,
            loader_routine=loader_routine,
        )
        trainer = cls(
            training_state=training_state,
            train_loader=train_loader,
            device=device,
            early_stopper=early_stopper,
            model_tracker=model_tracker,
            checkpoint_callback=checkpoint_callback,
            routine_suite=routine_suite,
        )
        return trainer

    @property
    def epoch_scores(self) -> pd.DataFrame:
        return self._epoch_score_cache.scores

    @property
    def latest_model_state(self) -> Dict[str, Any]:
        return self._training_state.model.state_dict()

    @property
    def best_model_state(self) -> Optional[Dict[str, Any]]:
        if self._model_tracker is not None:
            return self._model_tracker.best_model_state

    @staticmethod
    @abstractmethod
    def _get_optimizer(model: ModelTContr) -> optim.Optimizer: ...

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

    def _should_stop(self) -> bool:
        return self._early_stopper.stopped

    def _execute_epoch_preprocessing(self):
        self._routine_suite.attach_train_diagnostics_hooks(
            model=self._training_state.model, n_epochs_elapsed=self._n_epochs_elapsed
        )

    def _execute_epoch_postprocessing(self):
        super()._execute_epoch_postprocessing()
        self._execute_routine_suite()
        self._execute_checkpoint_callback()
        self._update_tracker()
        self._update_stopper()

    def _execute_routine_suite(self):
        self._routine_suite.clear_cache()
        self._routine_suite.execute(
            model=self._training_state.model, n_epochs_elapsed=self.n_epochs_elapsed
        )
        self._epoch_score_cache.append(self._routine_suite.scores)

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
                model=self._training_state.model,
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
