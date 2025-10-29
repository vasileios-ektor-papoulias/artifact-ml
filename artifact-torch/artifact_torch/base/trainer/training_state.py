from pathlib import Path
from typing import Any, Dict, Generic, Optional, TypeVar, Union

import torch
from torch import optim

from artifact_torch.base.model.base import Model

ModelT = TypeVar("ModelT", bound=Model[Any, Any])


class TrainingState(Generic[ModelT]):
    _model_state_checkpoint_key: str = "model_state"
    _optimizer_state_checkpoint_key: str = "optimizer_state"
    _scheduler_state_checkpoint_key: str = "scheduler_state"

    def __init__(
        self,
        model: ModelT,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
    ):
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._validate()

    @property
    def model(self) -> ModelT:
        return self._model

    @property
    def optimizer(self) -> optim.Optimizer:
        return self._optimizer

    @property
    def scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        return self._scheduler

    def serialize(self) -> Dict[str, Any]:
        checkpoint = {}
        checkpoint[self._model_state_checkpoint_key] = self._model.state_dict()
        checkpoint[self._optimizer_state_checkpoint_key] = self._optimizer.state_dict()
        if self._scheduler is not None:
            checkpoint[self._scheduler_state_checkpoint_key] = self._scheduler.state_dict()
        return checkpoint

    def export(self, path: Union[Path, str]):
        path = self._validate_checkpoint_path(path=path)
        checkpoint = self.serialize()
        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint: Dict[str, Any]):
        self._model.load_state_dict(checkpoint.get(self._model_state_checkpoint_key, {}))
        self._optimizer.load_state_dict(checkpoint.get(self._optimizer_state_checkpoint_key, {}))
        if self._scheduler is not None:
            self._scheduler.load_state_dict(
                checkpoint.get(self._scheduler_state_checkpoint_key, {})
            )

    def load(self, path: Union[Path, str], map_location: torch.device):
        checkpoint: Dict[str, Any] = torch.load(path, weights_only=True, map_location=map_location)
        self.load_checkpoint(checkpoint=checkpoint)

    def _validate(self):
        self._validate_optimizer(optimizer=self._optimizer, model=self._model)
        if self._scheduler is not None:
            self._validate_scheduler(
                scheduler=self._scheduler,
                optimizer=self._optimizer,
            )

    @staticmethod
    def _validate_optimizer(optimizer: optim.Optimizer, model: ModelT):
        model_param_ids = {id(p) for p in model.parameters()}
        optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        if not model_param_ids.issubset(optimizer_param_ids):
            raise ValueError("Optimizer not properly tied to model params.")

    @staticmethod
    def _validate_scheduler(scheduler: optim.lr_scheduler._LRScheduler, optimizer: optim.Optimizer):
        if scheduler.optimizer is not optimizer:
            raise ValueError("Scheduler not properly tied to optimizer.")

    @staticmethod
    def _validate_checkpoint_path(path: Union[str, Path]) -> Path:
        path = Path(path)
        if path.suffix != ".pth":
            raise ValueError(f"Invalid file extension: {path.suffix}. Expected '.pth'")
        return path
