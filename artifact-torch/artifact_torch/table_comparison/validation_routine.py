from abc import abstractmethod
from typing import Generic, List, Optional, Type, TypeVar

import pandas as pd
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison.artifacts.base import TableComparisonArtifactResources
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.table_comparison.validation_plan import TableComparisonValidationPlan

from artifact_torch.base.components.callbacks.data_loader import (
    DataLoaderArrayCallback,
    DataLoaderArrayCollectionCallback,
    DataLoaderPlotCallback,
    DataLoaderPlotCollectionCallback,
    DataLoaderScoreCallback,
    DataLoaderScoreCollectionCallback,
)
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.base.trainer.validation_routine import ValidationRoutine
from artifact_torch.core.model.generative import GenerationParams
from artifact_torch.table_comparison.model import TabularGenerativeModel
from artifact_torch.table_comparison.validation_plan_callback import TableComparisonPlanCallback

ModelInputT = TypeVar("ModelInputT", bound=ModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=ModelOutput)
GenerationParamsT = TypeVar("GenerationParamsT", bound=GenerationParams)
TableComparisonValidationRoutineT = TypeVar(
    "TableComparisonValidationRoutineT", bound="TableComparisonValidationRoutine"
)


class TableComparisonValidationRoutine(
    ValidationRoutine[
        TabularGenerativeModel[ModelInputT, ModelOutputT, GenerationParamsT],
        ModelInputT,
        ModelOutputT,
        TableComparisonValidationPlan,
        TableComparisonArtifactResources,
    ],
    Generic[ModelInputT, ModelOutputT, GenerationParamsT],
):
    @classmethod
    def build(
        cls: Type[TableComparisonValidationRoutineT],
        df_real: pd.DataFrame,
        tabular_data_spec: TabularDataSpecProtocol,
        train_loader: DataLoader[ModelInputT],
        val_loader: Optional[DataLoader[ModelInputT]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> TableComparisonValidationRoutineT:
        artifact_validation_period = cls._get_artifact_validation_period()
        generation_params = cls._get_generation_params()
        validation_plan = cls._get_validation_plan(
            tabular_data_spec=tabular_data_spec, tracking_client=tracking_client
        )
        validation_plan_callback = TableComparisonPlanCallback[GenerationParamsT](
            period=artifact_validation_period,
            validation_plan=validation_plan,
            df_real=df_real,
            generation_params=generation_params,
            tracking_client=tracking_client,
        )
        routine = super()._build(
            train_loader=train_loader,
            val_loader=val_loader,
            validation_plan_callback=validation_plan_callback,
            tracking_client=tracking_client,
        )
        return routine

    @staticmethod
    @abstractmethod
    def _get_generation_params() -> GenerationParamsT: ...

    @staticmethod
    @abstractmethod
    def _get_validation_plan(
        tabular_data_spec: TabularDataSpecProtocol,
        tracking_client: Optional[TrackingClient],
    ) -> TableComparisonValidationPlan: ...

    @staticmethod
    @abstractmethod
    def _get_artifact_validation_period() -> int: ...

    @staticmethod
    def _get_train_loader_score_callbacks() -> List[
        DataLoaderScoreCallback[ModelInputT, ModelOutputT]
    ]:
        return []

    @staticmethod
    def _get_train_loader_array_callbacks() -> List[
        DataLoaderArrayCallback[ModelInputT, ModelOutputT]
    ]:
        return []

    @staticmethod
    def _get_train_loader_plot_callbacks() -> List[
        DataLoaderPlotCallback[ModelInputT, ModelOutputT]
    ]:
        return []

    @staticmethod
    def _get_train_loader_score_collection_callbacks() -> List[
        DataLoaderScoreCollectionCallback[ModelInputT, ModelOutputT]
    ]:
        return []

    @staticmethod
    def _get_train_loader_array_collection_callbacks() -> List[
        DataLoaderArrayCollectionCallback[ModelInputT, ModelOutputT]
    ]:
        return []

    @staticmethod
    def _get_train_loader_plot_collection_callbacks() -> List[
        DataLoaderPlotCollectionCallback[ModelInputT, ModelOutputT]
    ]:
        return []

    @staticmethod
    def _get_val_loader_score_callbacks() -> List[
        DataLoaderScoreCallback[ModelInputT, ModelOutputT]
    ]:
        return []

    @staticmethod
    def _get_val_loader_array_callbacks() -> List[
        DataLoaderArrayCallback[ModelInputT, ModelOutputT]
    ]:
        return []

    @staticmethod
    def _get_val_loader_plot_callbacks() -> List[DataLoaderPlotCallback[ModelInputT, ModelOutputT]]:
        return []

    @staticmethod
    def _get_val_loader_score_collection_callbacks() -> List[
        DataLoaderScoreCollectionCallback[ModelInputT, ModelOutputT]
    ]:
        return []

    @staticmethod
    def _get_val_loader_array_collection_callbacks() -> List[
        DataLoaderArrayCollectionCallback[ModelInputT, ModelOutputT]
    ]:
        return []

    @staticmethod
    def _get_val_loader_plot_collection_callbacks() -> List[
        DataLoaderPlotCollectionCallback[ModelInputT, ModelOutputT]
    ]:
        return []
