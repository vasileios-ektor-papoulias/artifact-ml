from typing import Any, Mapping, Optional, Type

from artifact_core.table_comparison import TabularDataSpecProtocol
from artifact_experiment import DataSplit
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_torch.base.components.routines.loader import DataLoaderRoutine
from artifact_torch.base.components.routines.train_diagnostics import TrainDiagnosticsRoutine
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.base.trainer.trainer import Trainer
from artifact_torch.table_comparison.experiment import TabularSynthesisExperiment
from artifact_torch.table_comparison.model import TableSynthesizer
from artifact_torch.table_comparison.routine import (
    TableComparisonRoutine,
    TableComparisonRoutineData,
)

from demos.table_comparison.components.protocols import (
    DemoGenerationParams,
)
from demos.table_comparison.components.routines.artifact import DemoTableComparisonRoutine
from demos.table_comparison.components.routines.loader import DemoLoaderRoutine
from demos.table_comparison.components.routines.train_diagnostics import DemoTrainDiagnosticsRoutine
from demos.table_comparison.trainer.trainer import DemoTrainer


class DemoTabularSynthesisExperiment(
    TabularSynthesisExperiment[
        TableSynthesizer[Any, ModelOutput, DemoGenerationParams],
        ModelInput,
        ModelOutput,
        DemoGenerationParams,
    ]
):
    @classmethod
    def _get_trainer_type(
        cls,
    ) -> Type[
        Trainer[
            TableSynthesizer[Any, ModelOutput, DemoGenerationParams],
            ModelInput,
            ModelOutput,
            Any,
            Any,
        ]
    ]:
        return DemoTrainer

    @classmethod
    def _get_train_diagnostics_routine(
        cls,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[
        TrainDiagnosticsRoutine[
            TableSynthesizer[Any, ModelOutput, DemoGenerationParams],
            ModelInput,
            ModelOutput,
        ]
    ]:
        return DemoTrainDiagnosticsRoutine.build(tracking_client=tracking_client)

    @classmethod
    def _get_loader_routine(
        cls,
        data_loaders: Mapping[DataSplit, DataLoader[ModelInput]],
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[
        DataLoaderRoutine[
            TableSynthesizer[Any, ModelOutput, DemoGenerationParams],
            ModelInput,
            ModelOutput,
        ]
    ]:
        return DemoLoaderRoutine.build(data_loaders=data_loaders, tracking_client=tracking_client)

    @classmethod
    def _get_artifact_routine(
        cls,
        data: Mapping[DataSplit, TableComparisonRoutineData],
        data_spec: TabularDataSpecProtocol,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[TableComparisonRoutine[DemoGenerationParams]]:
        return DemoTableComparisonRoutine.build(
            data=data, data_spec=data_spec, tracking_client=tracking_client
        )
