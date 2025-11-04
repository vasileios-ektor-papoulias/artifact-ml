from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, Optional, Type, TypeVar

import pandas as pd
from artifact_core.base.artifact import ResourceSpecProtocol
from artifact_experiment.base.entities.data_split import DataSplit
from artifact_experiment.base.tracking.client import TrackingClient

from artifact_torch.base.components.routines.artifact import ArtifactRoutine, ArtifactRoutineData
from artifact_torch.base.components.routines.loader import DataLoaderRoutine
from artifact_torch.base.components.routines.train_diagnostics import TrainDiagnosticsRoutine
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.base.trainer.trainer import Trainer

ModelTContr = TypeVar("ModelTContr", bound=Model[Any, Any], contravariant=True)
ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
ArtifactRoutineDataTContr = TypeVar(
    "ArtifactRoutineDataTContr", bound=ArtifactRoutineData, contravariant=True
)
DataSpecProtocolTContr = TypeVar(
    "DataSpecProtocolTContr", bound=ResourceSpecProtocol, contravariant=True
)
ExperimentT = TypeVar("ExperimentT", bound="Experiment")


class Experiment(
    ABC,
    Generic[
        ModelTContr,
        ModelInputTContr,
        ModelOutputTContr,
        ArtifactRoutineDataTContr,
        DataSpecProtocolTContr,
    ],
):
    def __init__(
        self, trainer: Trainer[ModelTContr, ModelInputTContr, ModelOutputTContr, Any, Any]
    ):
        self._trainer = trainer

    @classmethod
    def build(
        cls: Type[ExperimentT],
        model: ModelTContr,
        data_loaders: Mapping[DataSplit, DataLoader[ModelInputTContr]],
        artifact_routine_data: Mapping[DataSplit, ArtifactRoutineDataTContr],
        artifact_routine_data_spec: Optional[DataSpecProtocolTContr] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> ExperimentT:
        assert DataSplit.TRAIN in data_loaders, "Training data not provided."
        train_diagnostics_routine = cls._get_train_diagnostics_routine(
            tracking_client=tracking_client
        )
        loader_routine = cls._get_loader_routine(
            data_loaders=data_loaders, tracking_client=tracking_client
        )
        artifact_routine = cls._get_artifact_routine(
            data=artifact_routine_data,
            data_spec=artifact_routine_data_spec,
            tracking_client=tracking_client,
        )
        trainer = cls._get_trainer_type().build(
            model=model,
            train_loader=data_loaders[DataSplit.TRAIN],
            train_diagnostics_routine=train_diagnostics_routine,
            loader_routine=loader_routine,
            artifact_routine=artifact_routine,
        )
        experiment = cls(trainer=trainer)
        return experiment

    @property
    def epoch_scores(self) -> pd.DataFrame:
        return self._trainer.epoch_scores

    @classmethod
    @abstractmethod
    def _get_trainer_type(
        cls,
    ) -> Type[Trainer[ModelTContr, ModelInputTContr, ModelOutputTContr, Any, Any]]: ...

    @classmethod
    @abstractmethod
    def _get_train_diagnostics_routine(
        cls,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[TrainDiagnosticsRoutine[ModelTContr, ModelInputTContr, ModelOutputTContr]]: ...

    @classmethod
    @abstractmethod
    def _get_loader_routine(
        cls,
        data_loaders: Mapping[DataSplit, DataLoader[ModelInputTContr]],
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[DataLoaderRoutine[ModelTContr, ModelInputTContr, ModelOutputTContr]]: ...

    @classmethod
    @abstractmethod
    def _get_artifact_routine(
        cls,
        data: Mapping[DataSplit, ArtifactRoutineDataTContr],
        data_spec: DataSpecProtocolTContr,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[ArtifactRoutine[ModelTContr, Any, Any, Any, Any]]: ...

    def run(self):
        self._trainer.train()
