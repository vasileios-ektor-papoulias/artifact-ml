from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Mapping, Optional, TypeVar

import pandas as pd
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison.artifacts.base import TableComparisonArtifactResources
from artifact_experiment import DataSplit
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.table_comparison.validation_plan import TableComparisonPlan

from artifact_torch.base.components.routines.artifact import (
    ArtifactRoutine,
    ArtifactRoutineData,
    ArtifactRoutineHyperparams,
)
from artifact_torch.core.model.generative import GenerationParams
from artifact_torch.libs.exports.table import TableExporter
from artifact_torch.table_comparison.model import TableSynthesizer

GenerationParamsTCov = TypeVar("GenerationParamsTCov", bound=GenerationParams, covariant=True)
TableComparisonRoutineT = TypeVar("TableComparisonRoutineT", bound="TableComparisonRoutine")


@dataclass
class TableComparisonRoutineHyperparams(ArtifactRoutineHyperparams, Generic[GenerationParamsTCov]):
    generation_params: GenerationParamsTCov


@dataclass
class TableComparisonRoutineData(ArtifactRoutineData):
    df_real: pd.DataFrame


class TableComparisonRoutine(
    ArtifactRoutine[
        TableSynthesizer[Any, Any, GenerationParamsTCov],
        TableComparisonRoutineHyperparams[GenerationParamsTCov],
        TableComparisonRoutineData,
        TableComparisonArtifactResources,
        TabularDataSpecProtocol,
    ],
    Generic[GenerationParamsTCov],
):
    _resource_export_prefix = "SYNTHETIC"

    @classmethod
    @abstractmethod
    def _get_periods(cls) -> Mapping[DataSplit, int]: ...

    @classmethod
    @abstractmethod
    def _get_validation_plans(
        cls,
        artifact_resource_spec: TabularDataSpecProtocol,
        tracking_client: Optional[TrackingClient],
    ) -> Mapping[DataSplit, TableComparisonPlan]: ...

    @classmethod
    @abstractmethod
    def _get_generation_params(cls) -> GenerationParamsTCov: ...

    @classmethod
    def _get_hyperparams(cls) -> TableComparisonRoutineHyperparams[GenerationParamsTCov]:
        generation_params = cls._get_generation_params()
        hyperparams = TableComparisonRoutineHyperparams[GenerationParamsTCov](
            generation_params=generation_params
        )
        return hyperparams

    def _generate_artifact_resources(
        self,
        model: TableSynthesizer[Any, Any, GenerationParamsTCov],
    ) -> Mapping[DataSplit, TableComparisonArtifactResources]:
        df_synthetic = model.generate(params=self._hyperparams.generation_params)
        resources_by_split = {}
        for data_split in self._data.keys():
            resources = TableComparisonArtifactResources(
                dataset_real=self._data[data_split].df_real, dataset_synthetic=df_synthetic
            )
            resources_by_split[data_split] = resources
        return resources_by_split

    @classmethod
    def _export_artifact_resources(
        cls,
        artifact_resources: TableComparisonArtifactResources,
        n_epochs_elapsed: int,
        data_split: DataSplit,
        tracking_client: TrackingClient,
    ):
        _ = data_split
        prefix = cls._resource_export_prefix
        TableExporter.export(
            data=artifact_resources.dataset_synthetic,
            tracking_client=tracking_client,
            prefix=prefix,
            step=n_epochs_elapsed,
        )
