from typing import Any, Optional, Type

from artifact_torch.nn import Trainer
from artifact_torch.nn.routines import DataLoaderRoutine, TrainDiagnosticsRoutine
from artifact_torch.table_comparison import (
    TableComparisonRoutine,
    TableSynthesizer,
    TabularSynthesisExperiment,
)

from demos.table_comparison.components.routines.artifact import DemoTableComparisonRoutine
from demos.table_comparison.components.routines.loader import DemoLoaderRoutine
from demos.table_comparison.components.routines.train_diagnostics import DemoTrainDiagnosticsRoutine
from demos.table_comparison.contracts.workflow import (
    WorkflowGenerationParams,
    WorkflowInput,
    WorkflowOutput,
)
from demos.table_comparison.trainer.trainer import DemoTrainer


class DemoTabularSynthesisExperiment(
    TabularSynthesisExperiment[
        TableSynthesizer[Any, WorkflowOutput, WorkflowGenerationParams],
        WorkflowInput,
        WorkflowOutput,
        WorkflowGenerationParams,
    ]
):
    @classmethod
    def _get_trainer(
        cls,
    ) -> Type[
        Trainer[
            TableSynthesizer[Any, WorkflowOutput, WorkflowGenerationParams],
            WorkflowInput,
            WorkflowOutput,
            Any,
            Any,
        ]
    ]:
        return DemoTrainer

    @classmethod
    def _get_train_diagnostics_routine(
        cls,
    ) -> Optional[
        Type[
            TrainDiagnosticsRoutine[
                TableSynthesizer[Any, WorkflowOutput, WorkflowGenerationParams],
                WorkflowInput,
                WorkflowOutput,
            ]
        ]
    ]:
        return DemoTrainDiagnosticsRoutine

    @classmethod
    def _get_loader_routine(
        cls,
    ) -> Optional[
        Type[
            DataLoaderRoutine[
                TableSynthesizer[Any, WorkflowOutput, WorkflowGenerationParams],
                WorkflowInput,
                WorkflowOutput,
            ]
        ]
    ]:
        return DemoLoaderRoutine

    @classmethod
    def _get_artifact_routine(
        cls,
    ) -> Optional[Type[TableComparisonRoutine[WorkflowGenerationParams]]]:
        return DemoTableComparisonRoutine
