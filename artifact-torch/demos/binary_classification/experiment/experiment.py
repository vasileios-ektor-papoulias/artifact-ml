from typing import Any, Mapping, Optional, Type

import pandas as pd
from artifact_core.binary_classification import BinaryFeatureSpecProtocol
from artifact_experiment import DataSplit
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_torch.base.components.routines.loader import DataLoaderRoutine
from artifact_torch.base.components.routines.train_diagnostics import TrainDiagnosticsRoutine
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.base.trainer.trainer import Trainer
from artifact_torch.binary_classification.experiment import BinaryClassificationExperiment
from artifact_torch.binary_classification.model import BinaryClassifier
from artifact_torch.binary_classification.routine import (
    BinaryClassificationRoutine,
    BinaryClassificationRoutineData,
)

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
    def _get_trainer_type(
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
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[
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
    ]:
        return DemoTrainDiagnosticsRoutine.build(tracking_client=tracking_client)

    @classmethod
    def _get_loader_routine(
        cls,
        data_loaders: Mapping[DataSplit, DataLoader[ModelInput]],
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[
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
    ]:
        return DemoLoaderRoutine.build(data_loaders=data_loaders, tracking_client=tracking_client)

    @classmethod
    def _get_artifact_routine(
        cls,
        data: Mapping[DataSplit, BinaryClassificationRoutineData[pd.DataFrame]],
        data_spec: BinaryFeatureSpecProtocol,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[BinaryClassificationRoutine[DemoClassificationParams, pd.DataFrame]]:
        return DemoBinaryClassificationRoutine.build(
            data=data, data_spec=data_spec, tracking_client=tracking_client
        )
