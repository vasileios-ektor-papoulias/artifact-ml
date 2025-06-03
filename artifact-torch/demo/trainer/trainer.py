from typing import Any, List, Optional

import torch
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_torch.base.components.callbacks.batch import (
    BatchCallback,
)
from artifact_torch.base.components.callbacks.checkpoint import CheckpointCallback
from artifact_torch.base.components.early_stopping.stopper import EarlyStopper, StopperUpdateData
from artifact_torch.base.components.logging.trainer_logger import TrainerLogger
from artifact_torch.base.components.model_tracking.tracker import (
    ModelTracker,
    ModelTrackingCriterion,
)
from artifact_torch.base.trainer.custom import CustomTrainer
from artifact_torch.libs.components.callbacks.batch.loss import BatchLossCallback
from artifact_torch.libs.components.early_stopping.epoch_bound import EpochBoundStopper
from torch import optim

from demo.config.constants import BATCH_LOSS_PERIOD, DEVICE, LEARNING_RATE, MAX_N_EPOCHS
from demo.model.io import TabularVAEInput, TabularVAEOutput
from demo.model.synthesizer import TabularVAESynthesizer
from demo.trainer.validation_routine import TabularVAEValidationRoutine


class TabularVAETrainer(
    CustomTrainer[
        TabularVAESynthesizer,
        TabularVAEInput,
        TabularVAEOutput,
        TabularVAEValidationRoutine,
        ModelTrackingCriterion,
        StopperUpdateData,
    ]
):
    @staticmethod
    def _get_optimizer(model: TabularVAESynthesizer) -> optim.Optimizer:
        return optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    @staticmethod
    def _get_scheduler(
        optimizer: optim.Optimizer,
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        _ = optimizer

    @staticmethod
    def _get_device() -> torch.device:
        return DEVICE

    @staticmethod
    def _get_model_tracker() -> Optional[ModelTracker[ModelTrackingCriterion]]:
        pass

    def _get_model_tracking_criterion(self) -> Optional[ModelTrackingCriterion]:
        pass

    @staticmethod
    def _get_early_stopper() -> EarlyStopper[StopperUpdateData]:
        return EpochBoundStopper(max_n_epochs=MAX_N_EPOCHS)

    def _get_stopper_update_data(self) -> StopperUpdateData:
        return StopperUpdateData(n_epochs_elapsed=self.n_epochs_elapsed)

    @staticmethod
    def _get_checkpoint_callback(
        tracking_client: Optional[TrackingClient],
    ) -> Optional[CheckpointCallback]:
        _ = tracking_client

    @staticmethod
    def _get_batch_callbacks(
        tracking_client: Optional[TrackingClient],
    ) -> List[BatchCallback[TabularVAEInput, TabularVAEOutput, Any]]:
        _ = tracking_client
        return [BatchLossCallback(period=BATCH_LOSS_PERIOD)]

    def _get_progress_bar_description(self) -> str:
        train_loss = self._epoch_score_cache.get_latest_non_null(key="train_loss")
        desc = TrainerLogger.get_progress_bar_desc(
            n_epochs_elapsed=self.n_epochs_elapsed, train_loss=train_loss
        )
        return desc
