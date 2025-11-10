from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, Optional, Type, TypeVar

import pandas as pd
from artifact_core.base.artifact import ResourceSpecProtocol
from artifact_experiment.base.entities.data_split import DataSplit
from artifact_experiment.base.tracking.backend.client import TrackingClient
from artifact_experiment.base.tracking.background.tracking_queue import TrackingQueue
from artifact_experiment.base.tracking.background.writer import FileWriter

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
        tracking_client: Optional[TrackingClient[Any]] = None,
    ) -> ExperimentT:
        assert DataSplit.TRAIN in data_loaders, "Training data not provided."
        tracking_queue = cls._get_tracking_queue(tracking_client=tracking_client)
        file_writer = cls._get_file_writer(tracking_client=tracking_client)
        train_diagnostics_routine = cls._build_train_diagnostics_routine(
            tracking_queue=tracking_queue
        )
        loader_routine = cls._build_loader_routine(
            data_loaders=data_loaders, tracking_queue=tracking_queue
        )
        artifact_routine = cls._build_artifact_routine(
            data=artifact_routine_data,
            data_spec=artifact_routine_data_spec,
            tracking_queue=tracking_queue,
        )
        trainer = cls._build_trainer(
            model=model,
            train_loader=data_loaders[DataSplit.TRAIN],
            train_diagnostics_routine=train_diagnostics_routine,
            loader_routine=loader_routine,
            artifact_routine=artifact_routine,
            file_writer=file_writer,
        )
        experiment = cls(trainer=trainer)
        return experiment

    @property
    def epoch_scores(self) -> pd.DataFrame:
        return self._trainer.epoch_scores

    @classmethod
    @abstractmethod
    def _get_trainer(
        cls,
    ) -> Type[Trainer[ModelTContr, ModelInputTContr, ModelOutputTContr, Any, Any]]: ...

    @classmethod
    @abstractmethod
    def _get_train_diagnostics_routine(
        cls,
    ) -> Optional[
        Type[TrainDiagnosticsRoutine[ModelTContr, ModelInputTContr, ModelOutputTContr]]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_loader_routine(
        cls,
    ) -> Optional[Type[DataLoaderRoutine[ModelTContr, ModelInputTContr, ModelOutputTContr]]]: ...

    @classmethod
    @abstractmethod
    def _get_artifact_routine(
        cls,
    ) -> Optional[
        Type[
            ArtifactRoutine[
                ModelTContr, Any, ArtifactRoutineDataTContr, Any, DataSpecProtocolTContr, Any
            ]
        ]
    ]: ...

    def run(self):
        self._trainer.train()

    @classmethod
    def _build_trainer(
        cls,
        model: ModelTContr,
        train_loader: DataLoader[ModelInputTContr],
        train_diagnostics_routine: Optional[
            TrainDiagnosticsRoutine[ModelTContr, ModelInputTContr, ModelOutputTContr]
        ],
        loader_routine: Optional[
            DataLoaderRoutine[ModelTContr, ModelInputTContr, ModelOutputTContr]
        ],
        artifact_routine: Optional[
            ArtifactRoutine[
                ModelTContr, Any, ArtifactRoutineDataTContr, Any, DataSpecProtocolTContr, Any
            ]
        ],
        file_writer: Optional[FileWriter],
    ) -> Trainer[ModelTContr, ModelInputTContr, ModelOutputTContr, Any, Any]:
        trainer_class = cls._get_trainer()
        trainer = trainer_class.build(
            model=model,
            train_loader=train_loader,
            train_diagnostics_routine=train_diagnostics_routine,
            loader_routine=loader_routine,
            artifact_routine=artifact_routine,
            file_writer=file_writer,
        )
        return trainer

    @classmethod
    def _build_train_diagnostics_routine(
        cls, tracking_queue: Optional[TrackingQueue]
    ) -> Optional[TrainDiagnosticsRoutine[ModelTContr, ModelInputTContr, ModelOutputTContr]]:
        routine = None
        routine_class = cls._get_train_diagnostics_routine()
        if routine_class is not None:
            routine = routine_class.build(tracking_queue=tracking_queue)
        return routine

    @classmethod
    def _build_loader_routine(
        cls,
        data_loaders: Mapping[DataSplit, DataLoader[ModelInputTContr]],
        tracking_queue: Optional[TrackingQueue],
    ) -> Optional[DataLoaderRoutine[ModelTContr, ModelInputTContr, ModelOutputTContr]]:
        routine = None
        routine_class = cls._get_loader_routine()
        if routine_class is not None:
            routine = routine_class.build(data_loaders=data_loaders, tracking_queue=tracking_queue)
        return routine

    @classmethod
    def _build_artifact_routine(
        cls,
        data: Mapping[DataSplit, ArtifactRoutineDataTContr],
        data_spec: Optional[DataSpecProtocolTContr],
        tracking_queue: Optional[TrackingQueue],
    ) -> Optional[
        ArtifactRoutine[
            ModelTContr, Any, ArtifactRoutineDataTContr, Any, DataSpecProtocolTContr, Any
        ]
    ]:
        routine = None
        routine_class = cls._get_artifact_routine()
        if routine_class is not None and data_spec is not None:
            routine = routine_class.build(
                data=data, data_spec=data_spec, tracking_queue=tracking_queue
            )
        return routine

    @staticmethod
    def _get_tracking_queue(tracking_client: Optional[TrackingClient]) -> Optional[TrackingQueue]:
        return tracking_client.queue if tracking_client is not None else None

    @staticmethod
    def _get_file_writer(tracking_client: Optional[TrackingClient]) -> Optional[FileWriter]:
        return tracking_client.file_writer if tracking_client is not None else None
