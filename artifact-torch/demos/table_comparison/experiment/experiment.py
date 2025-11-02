from typing import Any, Mapping, Optional, Type, TypeVar

from artifact_core.table_comparison import TabularDataSpecProtocol
from artifact_experiment import DataSplit
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_torch.base.components.routines.batch import BatchRoutine
from artifact_torch.base.components.routines.loader import DataLoaderRoutine
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.trainer.trainer import Trainer
from artifact_torch.table_comparison.experiment import TabularSynthesisExperiment
from artifact_torch.table_comparison.model import TableSynthesizer
from artifact_torch.table_comparison.routine import (
    TableComparisonRoutine,
    TableComparisonRoutineData,
)

from demos.table_comparison.components.routines.artifact import DemoTableComparisonRoutine
from demos.table_comparison.components.routines.batch import DemoBatchRoutine
from demos.table_comparison.components.routines.loader import DemoLoaderRoutine
from demos.table_comparison.components.routines.protocols import (
    DemoGenerationParams,
    DemoModelInput,
    DemoModelOutput,
)
from demos.table_comparison.trainer.trainer import DemoTrainer

ModelInputT = TypeVar("ModelInputT", bound=DemoModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=DemoModelOutput)


class DemoTabularSynthesisExperiment(
    TabularSynthesisExperiment[
        TableSynthesizer[ModelInputT, ModelOutputT, DemoGenerationParams],
        ModelInputT,
        ModelOutputT,
        DemoGenerationParams,
    ]
):
    @classmethod
    def _get_trainer_type(
        cls,
    ) -> Type[
        Trainer[
            TableSynthesizer[ModelInputT, ModelOutputT, DemoGenerationParams],
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
        data_loader: DataLoader[ModelInputT],
        data_split: DataSplit,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[
        DataLoaderRoutine[
            TableSynthesizer[ModelInputT, ModelOutputT, DemoGenerationParams],
            ModelInputT,
            ModelOutputT,
        ]
    ]:
        if data_split is DataSplit.TRAIN:
            return DemoLoaderRoutine.build(
                data_loader=data_loader, data_split=data_split, tracking_client=tracking_client
            )

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
