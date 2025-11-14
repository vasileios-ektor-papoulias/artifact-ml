from typing import Any, Optional

import torch
from artifact_experiment.base.tracking.background.writer import FileWriter
from artifact_torch.base.components.callbacks.checkpoint import CheckpointCallback
from artifact_torch.base.components.early_stopping.stopper import EarlyStopper, StopperUpdateData
from artifact_torch.base.components.model_tracking.tracker import (
    ModelTracker,
    ModelTrackingCriterion,
)
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.base.trainer.trainer import Trainer
from artifact_torch.libs.components.callbacks.export.checkpoint import TorchCheckpointCallback
from artifact_torch.libs.components.early_stopping.epoch_bound import EpochBoundStopper
from artifact_torch.table_comparison._model import TableSynthesizer
from torch import optim

from demos.table_comparison.config.constants import (
    CHECKPOINT_PERIOD,
    DEVICE,
    LEARNING_RATE,
    MAX_N_EPOCHS,
)


class DemoTrainer(
    Trainer[
        TableSynthesizer[Any, Any, Any],
        ModelInput,
        ModelOutput,
        StopperUpdateData,
        ModelTrackingCriterion,
    ]
):
    @staticmethod
    def _get_optimizer(
        model: TableSynthesizer[Any, Any, Any],
    ) -> optim.Optimizer:
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
        writer: Optional[FileWriter],
    ) -> Optional[CheckpointCallback]:
        if writer is not None:
            return TorchCheckpointCallback(period=CHECKPOINT_PERIOD, writer=writer)
