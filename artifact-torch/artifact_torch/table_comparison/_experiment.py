from abc import abstractmethod
from typing import Any, Generic, Optional, Type, TypeVar

from artifact_core.table_comparison.spi import TabularDataSpecProtocol

from artifact_torch._base.components.routines.loader import DataLoaderRoutine
from artifact_torch._base.components.routines.train_diagnostics import TrainDiagnosticsRoutine
from artifact_torch._base.experiment.experiment import Experiment
from artifact_torch._base.model.io import ModelInput, ModelOutput
from artifact_torch._base.trainer.trainer import Trainer
from artifact_torch._domains.generation.model import GenerationParams
from artifact_torch.table_comparison._model import TableSynthesizer
from artifact_torch.table_comparison._routine import (
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
    def _get_trainer(
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
    ) -> Optional[
        Type[TrainDiagnosticsRoutine[TableSynthesizerTContr, ModelInputTContr, ModelOutputTContr]]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_loader_routine(
        cls,
    ) -> Optional[
        Type[DataLoaderRoutine[TableSynthesizerTContr, ModelInputTContr, ModelOutputTContr]]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_artifact_routine(
        cls,
    ) -> Optional[Type[TableComparisonRoutine[GenerationParamsTContr]]]: ...
