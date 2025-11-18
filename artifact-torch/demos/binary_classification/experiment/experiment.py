from typing import Any, Optional, Type

import pandas as pd
from artifact_torch.binary_classification import (
    BinaryClassificationExperiment,
    BinaryClassificationRoutine,
    BinaryClassifier,
)
from artifact_torch.nn import Trainer
from artifact_torch.nn.routines import DataLoaderRoutine, TrainDiagnosticsRoutine

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
            WorkflowOutput,
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
                WorkflowOutput,
                WorkflowClassificationParams,
                pd.DataFrame,
            ],
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
                BinaryClassifier[
                    Any,
                    WorkflowOutput,
                    WorkflowClassificationParams,
                    pd.DataFrame,
                ],
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
                BinaryClassifier[
                    Any,
                    WorkflowOutput,
                    WorkflowClassificationParams,
                    pd.DataFrame,
                ],
                WorkflowInput,
                WorkflowOutput,
            ]
        ]
    ]:
        return DemoLoaderRoutine

    @classmethod
    def _get_artifact_routine(
        cls,
    ) -> Optional[Type[BinaryClassificationRoutine[WorkflowClassificationParams, pd.DataFrame]]]:
        return DemoBinaryClassificationRoutine
