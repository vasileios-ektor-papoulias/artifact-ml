from abc import abstractmethod
from typing import Any, Generic, List, Optional, Type, TypeVar

import torch
from artifact_experiment.base.tracking.client import TrackingClient
from torch import optim

from artifact_torch.base.components.callbacks.batch import (
    BatchCallback,
)
from artifact_torch.base.components.callbacks.checkpoint import (
    CheckpointCallback,
)
from artifact_torch.base.components.early_stopping.stopper import EarlyStopper, StopperUpdateData
from artifact_torch.base.components.model_tracking.tracker import (
    ModelTracker,
    ModelTrackingCriterion,
)
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.base.trainer.custom import CustomTrainer
from artifact_torch.base.trainer.validation_routine import ValidationRoutine

ModelInputT = TypeVar("ModelInputT", bound=ModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=ModelOutput)
ModelT = TypeVar("ModelT", bound=Model[Any, Any])
ValidationRoutineT = TypeVar("ValidationRoutineT", bound=ValidationRoutine)
ModelTrackingCriterionT = TypeVar("ModelTrackingCriterionT", bound=ModelTrackingCriterion)
StopperUpdateDataT = TypeVar("StopperUpdateDataT", bound=StopperUpdateData)
TrainerT = TypeVar("TrainerT", bound="CustomTrainer")


class Trainer(
    CustomTrainer[
        ModelT,
        ModelInputT,
        ModelOutputT,
        ValidationRoutineT,
        ModelTrackingCriterionT,
        StopperUpdateDataT,
    ],
    Generic[
        ModelT,
        ModelInputT,
        ModelOutputT,
        ValidationRoutineT,
        ModelTrackingCriterionT,
        StopperUpdateDataT,
    ],
):
    @classmethod
    def build(
        cls: Type[TrainerT],
        model: ModelT,
        train_loader: DataLoader[ModelInputT],
        validation_routine: ValidationRoutineT,
        tracking_client: Optional[TrackingClient] = None,
    ) -> TrainerT:
        trainer = super()._build(
            model=model,
            train_loader=train_loader,
            validation_routine=validation_routine,
            tracking_client=tracking_client,
        )
        return trainer

    @staticmethod
    @abstractmethod
    def _get_optimizer(model: ModelT) -> optim.Optimizer: ...

    @staticmethod
    @abstractmethod
    def _get_early_stopper() -> EarlyStopper[StopperUpdateDataT]: ...

    @abstractmethod
    def _get_stopper_update_data(self) -> StopperUpdateDataT: ...

    @staticmethod
    def _get_scheduler(
        optimizer: optim.Optimizer,
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        _ = optimizer

    @staticmethod
    def _get_device() -> torch.device:
        return torch.device("cpu")

    @staticmethod
    def _get_model_tracker() -> Optional[ModelTracker]:
        pass

    def _get_model_tracking_criterion(self) -> Optional[ModelTrackingCriterionT]:
        pass

    @staticmethod
    def _get_checkpoint_callback(
        tracking_client: Optional[TrackingClient],
    ) -> Optional[CheckpointCallback]:
        _ = tracking_client

    @staticmethod
    def _get_batch_callbacks(
        tracking_client: Optional[TrackingClient],
    ) -> List[BatchCallback[ModelInputT, ModelOutputT, Any]]:
        return []
