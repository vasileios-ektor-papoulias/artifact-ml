from abc import abstractmethod
from typing import Any, Generic, Mapping, Optional, Type, TypeVar

from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_experiment.base.data_split import DataSplit
from artifact_experiment.base.tracking.client import TrackingClient

from artifact_torch.base.components.routines.batch import BatchRoutine
from artifact_torch.base.components.routines.loader import DataLoaderRoutine
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.experiment.experiment import Experiment
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.base.trainer.trainer import Trainer
from artifact_torch.core.model.generative import GenerationParams
from artifact_torch.table_comparison.model import TableSynthesizer
from artifact_torch.table_comparison.routine import (
    TableComparisonRoutine,
    TableComparisonRoutineData,
)

ModelInputT = TypeVar("ModelInputT", bound=ModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=ModelOutput)
TableSynthesizerT = TypeVar("TableSynthesizerT", bound=TableSynthesizer[Any, Any, Any])
GenerationParamsT = TypeVar("GenerationParamsT", bound=GenerationParams)
GenericTabularSynthesisExperimentT = TypeVar(
    "GenericTabularSynthesisExperimentT", bound="GenericTabularSynthesisExperiment"
)


class GenericTabularSynthesisExperiment(
    Experiment[
        TableSynthesizerT,
        ModelInputT,
        ModelOutputT,
        TabularDataSpecProtocol,
        TableComparisonRoutineData,
    ],
    Generic[TableSynthesizerT, ModelInputT, ModelOutputT, GenerationParamsT],
):
    @classmethod
    @abstractmethod
    def _get_trainer_type(
        cls,
    ) -> Type[
        Trainer[
            TableSynthesizerT,
            ModelInputT,
            ModelOutputT,
            Any,
            Any,
        ]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_batch_routine(
        cls,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[
        BatchRoutine[
            ModelInputT,
            ModelOutputT,
            TableSynthesizerT,
        ]
    ]: ...

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
        data: Mapping[DataSplit, TableComparisonRoutineData],
        data_spec: TabularDataSpecProtocol,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[TableComparisonRoutine[GenerationParamsT]]: ...


TabularSynthesisExperimentT = TypeVar(
    "TabularSynthesisExperimentT", bound="TabularSynthesisExperiment"
)


class TabularSynthesisExperiment(
    GenericTabularSynthesisExperiment[
        TableSynthesizer[ModelInputT, ModelOutputT, GenerationParamsT],
        ModelInputT,
        ModelOutputT,
        GenerationParamsT,
    ],
    Generic[ModelInputT, ModelOutputT, GenerationParamsT],
):
    @classmethod
    @abstractmethod
    def _get_trainer_type(
        cls,
    ) -> Type[
        Trainer[
            TableSynthesizer[ModelInputT, ModelOutputT, GenerationParamsT],
            ModelInputT,
            ModelOutputT,
            Any,
            Any,
        ]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_batch_routine(
        cls,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[
        BatchRoutine[
            ModelInputT,
            ModelOutputT,
            TableSynthesizer[ModelInputT, ModelOutputT, GenerationParamsT],
        ]
    ]: ...

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
        data: Mapping[DataSplit, TableComparisonRoutineData],
        data_spec: TabularDataSpecProtocol,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[TableComparisonRoutine[GenerationParamsT]]: ...
