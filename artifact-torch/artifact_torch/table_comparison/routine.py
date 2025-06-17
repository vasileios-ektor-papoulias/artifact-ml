from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Optional, Type, TypeVar

import pandas as pd
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison.artifacts.base import TableComparisonArtifactResources
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.table_comparison.validation_plan import TableComparisonPlan

from artifact_torch.base.components.routines.artifact import (
    ArtifactRoutineData,
    ArtifactRoutineHyperparams,
    ArtifactValidationRoutine,
)
from artifact_torch.core.model.generative import GenerationParams
from artifact_torch.libs.exports.table import TableExporter
from artifact_torch.table_comparison.model import TableSynthesizer

GenerationParamsT = TypeVar("GenerationParamsT", bound=GenerationParams)
TableComparisonRoutineT = TypeVar("TableComparisonRoutineT", bound="TableComparisonRoutine")


@dataclass
class TableComparisonRoutineHyperparams(ArtifactRoutineHyperparams, Generic[GenerationParamsT]):
    generation_params: GenerationParamsT


@dataclass
class TableComparisonRoutineData(ArtifactRoutineData):
    df_real: pd.DataFrame


class TableComparisonRoutine(
    ArtifactValidationRoutine[
        TableSynthesizer[Any, Any, GenerationParamsT],
        TableComparisonRoutineHyperparams[GenerationParamsT],
        TableComparisonRoutineData,
        TableComparisonArtifactResources,
        TabularDataSpecProtocol,
    ],
    Generic[GenerationParamsT],
):
    _resource_export_prefix = "synthetic"

    @classmethod
    def build(
        cls: Type[TableComparisonRoutineT],
        df_real: pd.DataFrame,
        data_spec: TabularDataSpecProtocol,
        tracking_client: Optional[TrackingClient] = None,
    ) -> TableComparisonRoutineT:
        data = TableComparisonRoutineData(df_real=df_real)
        routine = cls._build(
            data=data,
            artifact_resource_spec=data_spec,
            tracking_client=tracking_client,
        )
        return routine

    @classmethod
    @abstractmethod
    def _get_period(cls) -> int: ...

    @classmethod
    @abstractmethod
    def _get_generation_params(cls) -> GenerationParamsT: ...

    @classmethod
    @abstractmethod
    def _get_validation_plan(
        cls,
        artifact_resource_spec: TabularDataSpecProtocol,
        tracking_client: Optional[TrackingClient],
    ) -> TableComparisonPlan: ...

    @classmethod
    def _get_hyperparams(cls) -> TableComparisonRoutineHyperparams[GenerationParamsT]:
        generation_params = cls._get_generation_params()
        hyperparams = TableComparisonRoutineHyperparams[GenerationParamsT](
            generation_params=generation_params
        )
        return hyperparams

    @classmethod
    def _generate_artifact_resources(
        cls,
        model: TableSynthesizer[Any, Any, GenerationParamsT],
        hyperparams: TableComparisonRoutineHyperparams,
        data: TableComparisonRoutineData,
    ) -> TableComparisonArtifactResources:
        df_synthetic = model.generate(params=hyperparams.generation_params)
        resources = TableComparisonArtifactResources(
            dataset_real=data.df_real, dataset_synthetic=df_synthetic
        )
        return resources

    @classmethod
    def _export_artifact_resources(
        cls,
        artifact_resources: TableComparisonArtifactResources,
        n_epochs_elapsed: int,
        tracking_client: TrackingClient,
    ):
        TableExporter.export(
            data=artifact_resources.dataset_synthetic,
            tracking_client=tracking_client,
            prefix=cls._resource_export_prefix,
            step=n_epochs_elapsed,
        )
