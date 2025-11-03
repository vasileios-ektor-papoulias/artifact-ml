from typing import Any, Mapping, Optional, Type, TypeVar

import pandas as pd
from artifact_core.binary_classification import BinaryFeatureSpecProtocol
from artifact_experiment import DataSplit
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_torch.base.components.routines.batch import BatchRoutine
from artifact_torch.base.components.routines.loader import DataLoaderRoutine
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.trainer.trainer import Trainer
from artifact_torch.binary_classification.experiment import BinaryClassificationExperiment
from artifact_torch.binary_classification.model import BinaryClassifier
from artifact_torch.binary_classification.routine import (
    BinaryClassificationRoutine,
    BinaryClassificationRoutineData,
)

from demos.binary_classification.components.protocols import (
    DemoClassificationParams,
    DemoModelInput,
    DemoModelOutput,
)
from demos.binary_classification.components.routines.artifact import DemoBinaryClassificationRoutine
from demos.binary_classification.components.routines.batch import DemoBatchRoutine
from demos.binary_classification.components.routines.loader import DemoLoaderRoutine
from demos.binary_classification.trainer.trainer import DemoTrainer

ModelInputT = TypeVar("ModelInputT", bound=DemoModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=DemoModelOutput)


class DemoBinaryClassificationExperiment(
    BinaryClassificationExperiment[
        BinaryClassifier[
            ModelInputT,
            ModelOutputT,
            DemoClassificationParams,
            pd.DataFrame,
        ],
        ModelInputT,
        ModelOutputT,
        DemoClassificationParams,
        pd.DataFrame,
    ]
):
    @classmethod
    def _get_trainer_type(
        cls,
    ) -> Type[
        Trainer[
            BinaryClassifier[ModelInputT, ModelOutputT, DemoClassificationParams, pd.DataFrame],
            ModelInputT,
            ModelOutputT,
            Any,
            Any,
        ]
    ]:
        return DemoTrainer[ModelInputT, ModelOutputT]

    @classmethod
    def _get_batch_routine(
        cls,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[BatchRoutine[ModelInputT, ModelOutputT]]:
        return DemoBatchRoutine.build(tracking_client=tracking_client)

    @classmethod
    def _get_loader_routine(
        cls,
        data_loaders: Mapping[DataSplit, DataLoader[ModelInputT]],
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[
        DataLoaderRoutine[
            BinaryClassifier[
                ModelInputT,
                ModelOutputT,
                DemoClassificationParams,
                pd.DataFrame,
            ],
            ModelInputT,
            ModelOutputT,
        ]
    ]:
        return DemoLoaderRoutine.build(data_loaders=data_loaders, tracking_client=tracking_client)

    @classmethod
    def _get_artifact_routine(
        cls,
        data: Mapping[DataSplit, BinaryClassificationRoutineData],
        data_spec: BinaryFeatureSpecProtocol,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[BinaryClassificationRoutine[DemoClassificationParams, pd.DataFrame]]:
        return DemoBinaryClassificationRoutine.build(
            data=data, data_spec=data_spec, tracking_client=tracking_client
        )
