from typing import Any, Optional, Type

import pandas as pd
from artifact_torch.binary_classification._experiment import BinaryClassificationExperiment
from artifact_torch.binary_classification._model import BinaryClassifier
from artifact_torch.binary_classification._routine import BinaryClassificationRoutine
from artifact_torch.core import ModelInput, ModelOutput, Trainer
from artifact_torch.routines import DataLoaderRoutine, TrainDiagnosticsRoutine

from demos.binary_classification.components.routines.artifact import DemoBinaryClassificationRoutine
from demos.binary_classification.components.routines.loader import DemoLoaderRoutine
from demos.binary_classification.components.routines.train_diagnostics import (
    DemoTrainDiagnosticsRoutine,
)
from demos.binary_classification.contracts.workflow import (
    WorkflowClassificationParams,
    WorkflowInput,
    WorkflowOutput,
)
from demos.binary_classification.trainer.trainer import DemoTrainer


class DemoBinaryClassificationExperiment(
    BinaryClassificationExperiment[
        BinaryClassifier[
            Any,
            ModelOutput,
            WorkflowClassificationParams,
            pd.DataFrame,
        ],
        WorkflowInput,
        WorkflowOutput,
        WorkflowClassificationParams,
        pd.DataFrame,
    ]
):
    @classmethod
    def _get_trainer(
        cls,
    ) -> Type[
        Trainer[
            BinaryClassifier[
                Any,
                ModelOutput,
                WorkflowClassificationParams,
                pd.DataFrame,
            ],
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
                BinaryClassifier[
                    Any,
                    ModelOutput,
                    WorkflowClassificationParams,
                    pd.DataFrame,
                ],
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
                BinaryClassifier[
                    Any,
                    ModelOutput,
                    WorkflowClassificationParams,
                    pd.DataFrame,
                ],
                ModelInput,
                ModelOutput,
            ]
        ]
    ]:
        return DemoLoaderRoutine

    @classmethod
    def _get_artifact_routine(
        cls,
    ) -> Optional[Type[BinaryClassificationRoutine[WorkflowClassificationParams, pd.DataFrame]]]:
        return DemoBinaryClassificationRoutine
