from typing import Any, Generic, Optional, TypeVar

import torch
from artifact_experiment.tracking import TrackingClient
from artifact_torch.base.components.callbacks.checkpoint import CheckpointCallback
from artifact_torch.base.components.early_stopping.stopper import EarlyStopper, StopperUpdateData
from artifact_torch.base.components.model_tracking.tracker import (
    ModelTracker,
    ModelTrackingCriterion,
)
from artifact_torch.base.trainer.trainer import Trainer
from artifact_torch.libs.components.callbacks.checkpoint.standard import StandardCheckpointCallback
from artifact_torch.libs.components.early_stopping.epoch_bound import EpochBoundStopper
from artifact_torch.table_comparison.model import TableSynthesizer
from torch import optim

from demos.table_comparison.components.routines.protocols import (
    DemoModelInput,
    DemoModelOutput,
)
from demos.table_comparison.config.constants import (
    CHECKPOINT_PERIOD,
    DEVICE,
    LEARNING_RATE,
    MAX_N_EPOCHS,
)

ModelInputT = TypeVar("ModelInputT", bound=DemoModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=DemoModelOutput)


class DemoTrainer(
    Trainer[
        TableSynthesizer[ModelInputT, ModelOutputT, Any],
        ModelInputT,
        ModelOutputT,
        StopperUpdateData,
        ModelTrackingCriterion,
    ],
    Generic[ModelInputT, ModelOutputT],
):
    @staticmethod
    def _get_optimizer(
        model: TableSynthesizer[ModelInputT, ModelOutputT, Any],
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
        tracking_client: Optional[TrackingClient],
    ) -> Optional[CheckpointCallback]:
        if tracking_client is not None:
            return StandardCheckpointCallback(
                period=CHECKPOINT_PERIOD, tracking_client=tracking_client
            )
