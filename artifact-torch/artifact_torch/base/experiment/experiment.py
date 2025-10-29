from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, Optional, Type, TypeVar

import pandas as pd
from artifact_core.base.artifact import ResourceSpecProtocol
from artifact_experiment.base.data_split import DataSplit
from artifact_experiment.base.tracking.client import TrackingClient

from artifact_torch.base.components.routines.artifact import ArtifactRoutine, ArtifactRoutineData
from artifact_torch.base.components.routines.batch import BatchRoutine
from artifact_torch.base.components.routines.loader import DataLoaderRoutine
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.base.trainer.trainer import Trainer

ModelT = TypeVar("ModelT", bound=Model[Any, Any])
ModelInputT = TypeVar("ModelInputT", bound=ModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=ModelOutput)
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ArtifactRoutineDataT = TypeVar("ArtifactRoutineDataT", bound=ArtifactRoutineData)
ExperimentT = TypeVar("ExperimentT", bound="Experiment")


class Experiment(
    ABC,
    Generic[ModelT, ModelInputT, ModelOutputT, ResourceSpecProtocolT, ArtifactRoutineDataT],
):
    def __init__(self, trainer: Trainer[ModelT, ModelInputT, ModelOutputT, Any, Any]):
        self._trainer = trainer

    @classmethod
    def build(
        cls: Type[ExperimentT],
        model: ModelT,
        data_loaders: Mapping[DataSplit, DataLoader[ModelInputT]],
        artifact_routine_data: Mapping[DataSplit, ArtifactRoutineDataT],
        artifact_routine_data_spec: Optional[ResourceSpecProtocolT] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> ExperimentT:
        assert DataSplit.TRAIN in data_loaders, "Training data not provided."
        batch_routine = cls._get_batch_routine(tracking_client=tracking_client)
        loader_routines = [
            loader_routine
            for loader_routine in [
                cls._get_loader_routine(
                    data_loader=data_loader, data_split=data_split, tracking_client=tracking_client
                )
                for data_split, data_loader in data_loaders.items()
            ]
            if loader_routine is not None
        ]
        artifact_routine = (
            cls._get_artifact_routine(
                data=artifact_routine_data,
                data_spec=artifact_routine_data_spec,
                tracking_client=tracking_client,
            )
            if artifact_routine_data_spec is not None
            else None
        )
        trainer = cls._get_trainer_type().build(
            model=model,
            train_loader=data_loaders[DataSplit.TRAIN],
            batch_routine=batch_routine,
            loader_routines=loader_routines,
            artifact_routine=artifact_routine,
        )
        experiment = cls(trainer=trainer)
        return experiment

    @property
    def model(self) -> ModelT:
        return self._trainer.model

    @property
    def epoch_scores(self) -> pd.DataFrame:
        return self._trainer.epoch_scores

    @classmethod
    @abstractmethod
    def _get_trainer_type(
        cls,
    ) -> Type[Trainer[ModelT, ModelInputT, ModelOutputT, Any, Any]]: ...

    @classmethod
    @abstractmethod
    def _get_batch_routine(
        cls,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[BatchRoutine[ModelInputT, ModelOutputT]]: ...

    @classmethod
    @abstractmethod
    def _get_loader_routine(
        cls,
        data_loader: DataLoader[ModelInputT],
        data_split: DataSplit,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[DataLoaderRoutine[ModelInputT, ModelOutputT]]: ...

    @classmethod
    @abstractmethod
    def _get_artifact_routine(
        cls,
        data: Mapping[DataSplit, ArtifactRoutineDataT],
        data_spec: ResourceSpecProtocolT,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[ArtifactRoutine[ModelT, Any, Any, Any, Any]]: ...

    def run(self):
        self._trainer.train()
