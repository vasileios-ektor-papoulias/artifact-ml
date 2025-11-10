from typing import Any, Optional, Type

import pandas as pd
from artifact_torch.base.components.routines.loader import DataLoaderRoutine
from artifact_torch.base.components.routines.train_diagnostics import TrainDiagnosticsRoutine
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.base.trainer.trainer import Trainer
from artifact_torch.binary_classification.experiment import BinaryClassificationExperiment
from artifact_torch.binary_classification.model import BinaryClassifier
from artifact_torch.binary_classification.routine import BinaryClassificationRoutine

from demos.binary_classification.components.protocols import DemoClassificationParams
from demos.binary_classification.components.routines.artifact import DemoBinaryClassificationRoutine
from demos.binary_classification.components.routines.loader import DemoLoaderRoutine
from demos.binary_classification.components.routines.train_diagnostics import (
    DemoTrainDiagnosticsRoutine,
)
from demos.binary_classification.trainer.trainer import DemoTrainer


class DemoBinaryClassificationExperiment(
    BinaryClassificationExperiment[
        BinaryClassifier[
            Any,
            ModelOutput,
            DemoClassificationParams,
            pd.DataFrame,
        ],
        ModelInput,
        ModelOutput,
        DemoClassificationParams,
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
                DemoClassificationParams,
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
                    DemoClassificationParams,
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
                    DemoClassificationParams,
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
    ) -> Optional[Type[BinaryClassificationRoutine[DemoClassificationParams, pd.DataFrame]]]:
        return DemoBinaryClassificationRoutine
