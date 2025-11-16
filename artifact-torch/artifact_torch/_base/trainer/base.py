from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, Optional, TypeVar

import torch
from torch import optim
from tqdm import tqdm

from artifact_torch._base.data.data_loader import DataLoader
from artifact_torch._base.model.base import Model
from artifact_torch._base.model.io import ModelInput, ModelOutput
from artifact_torch._base.trainer.training_state import TrainingState

ModelTContr = TypeVar("ModelTContr", bound=Model[Any, Any], contravariant=True)
ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)


class TrainerBase(ABC, Generic[ModelTContr, ModelInputTContr, ModelOutputTContr]):
    def __init__(
        self,
        training_state: TrainingState[ModelTContr],
        train_loader: DataLoader[ModelInputTContr],
        device: torch.device,
    ):
        self._training_state = training_state
        self._train_loader = train_loader
        self._device = device
        self._n_epochs_elapsed = 0

    @property
    def training_state(self) -> TrainingState[ModelTContr]:
        return self._training_state

    @property
    def train_loader(self) -> DataLoader[ModelInputTContr]:
        return self._train_loader

    @property
    def optimizer(self) -> optim.Optimizer:
        return self._training_state.optimizer

    @property
    def scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        return self._training_state.scheduler

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self._device = device

    @property
    def n_epochs_elapsed(self) -> int:
        return self._n_epochs_elapsed

    @abstractmethod
    def _should_stop(self) -> bool: ...

    @abstractmethod
    def _execute_epoch_preprocessing(self): ...

    @abstractmethod
    def _execute_epoch_postprocessing(self):
        self._n_epochs_elapsed += 1

    @abstractmethod
    def _get_progress_bar_description(self) -> str: ...

    def train(self):
        self._training_state.model.device = self.device
        print(f"Training on device: {self.device}")
        while not self._should_stop():
            self._execute_epoch_preprocessing()
            self._train_one_epoch()
            self._execute_epoch_postprocessing()

    def load_checkpoint(self, path: Path):
        self._training_state.load(path=path, map_location=self.device)

    def _train_one_epoch(self):
        self._training_state.model.train()
        self._train_loader.device = self._training_state.model.device
        progress_bar_desc = self._get_progress_bar_description()
        for batch in tqdm(self._train_loader, desc=progress_bar_desc, leave=False):
            self._train_one_batch(batch=batch)
        self._training_state.model.eval()

    def _train_one_batch(self, batch: ModelInputTContr):
        model_output = self._training_state.model(batch)
        t_loss = self._get_loss(model_output=model_output)
        self._training_step(
            t_loss=t_loss,
            optimizer=self._training_state.optimizer,
            scheduler=self._training_state.scheduler,
        )
        return model_output

    @staticmethod
    def _training_step(
        t_loss: torch.Tensor,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ):
        optimizer.zero_grad()
        t_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    @staticmethod
    def _get_loss(model_output: ModelOutputTContr) -> torch.Tensor:
        loss = model_output.get("t_loss")
        assert loss is not None
        return loss

    def _get_checkpoint(self) -> Dict[str, Any]:
        return self._training_state.serialize()
