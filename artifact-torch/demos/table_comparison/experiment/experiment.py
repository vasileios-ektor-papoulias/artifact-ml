from typing import Any, Optional, Type

from artifact_torch.base.components.routines.loader import DataLoaderRoutine
from artifact_torch.base.components.routines.train_diagnostics import TrainDiagnosticsRoutine
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.base.trainer.trainer import Trainer
from artifact_torch.table_comparison._experiment import TabularSynthesisExperiment
from artifact_torch.table_comparison._model import TableSynthesizer
from artifact_torch.table_comparison._routine import TableComparisonRoutine

from demos.table_comparison.components.protocols import DemoGenerationParams
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
    def _get_trainer(
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
    ) -> Optional[
        Type[
            TrainDiagnosticsRoutine[
                TableSynthesizer[Any, ModelOutput, DemoGenerationParams],
                ModelInput,
                ModelOutput,
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
                TableSynthesizer[Any, ModelOutput, DemoGenerationParams],
                ModelInput,
                ModelOutput,
            ]
        ]
    ]:
        return DemoLoaderRoutine

    @classmethod
    def _get_artifact_routine(cls) -> Optional[Type[TableComparisonRoutine[DemoGenerationParams]]]:
        return DemoTableComparisonRoutine
