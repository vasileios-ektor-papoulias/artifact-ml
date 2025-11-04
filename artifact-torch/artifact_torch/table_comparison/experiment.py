from abc import abstractmethod
from typing import Any, Generic, Mapping, Optional, Type, TypeVar

from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_experiment.base.entities.data_split import DataSplit
from artifact_experiment.base.tracking.client import TrackingClient

from artifact_torch.base.components.routines.loader import DataLoaderRoutine
from artifact_torch.base.components.routines.train_diagnostics import TrainDiagnosticsRoutine
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

TableSynthesizerTContr = TypeVar(
    "TableSynthesizerTContr", bound=TableSynthesizer[Any, Any, Any], contravariant=True
)
ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
GenerationParamsTContr = TypeVar(
    "GenerationParamsTContr", bound=GenerationParams, contravariant=True
)
TabularSynthesisExperimentT = TypeVar(
    "TabularSynthesisExperimentT", bound="TabularSynthesisExperiment"
)


class TabularSynthesisExperiment(
    Experiment[
        TableSynthesizerTContr,
        ModelInputTContr,
        ModelOutputTContr,
        TableComparisonRoutineData,
        TabularDataSpecProtocol,
    ],
    Generic[TableSynthesizerTContr, ModelInputTContr, ModelOutputTContr, GenerationParamsTContr],
):
    @classmethod
    @abstractmethod
    def _get_trainer_type(
        cls,
    ) -> Type[
        Trainer[
            TableSynthesizerTContr,
            ModelInputTContr,
            ModelOutputTContr,
            Any,
            Any,
        ]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_train_diagnostics_routine(
        cls,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[
        TrainDiagnosticsRoutine[TableSynthesizerTContr, ModelInputTContr, ModelOutputTContr]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_loader_routine(
        cls,
        data_loaders: Mapping[DataSplit, DataLoader[ModelInputTContr]],
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[
        DataLoaderRoutine[TableSynthesizerTContr, ModelInputTContr, ModelOutputTContr]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_artifact_routine(
        cls,
        data: Mapping[DataSplit, TableComparisonRoutineData],
        data_spec: TabularDataSpecProtocol,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[TableComparisonRoutine[GenerationParamsTContr]]: ...
