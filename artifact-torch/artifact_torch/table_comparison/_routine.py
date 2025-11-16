from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Mapping, Optional, Type, TypeVar

import pandas as pd
from artifact_core.table_comparison._artifacts.base import TableComparisonArtifactResources
from artifact_core.table_comparison.spi import TabularDataSpecProtocol
from artifact_experiment.table_comparison import TableComparisonPlan
from artifact_experiment.tracking import DataSplit

from artifact_torch._base.components.routines.artifact import (
    ArtifactRoutine,
    ArtifactRoutineData,
    ArtifactRoutineHyperparams,
)
from artifact_torch._domains.generation.model import GenerationParams
from artifact_torch.table_comparison._model import TableSynthesizer

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
        TableComparisonRoutineData,
        TableComparisonRoutineHyperparams[GenerationParamsTCov],
        TableComparisonArtifactResources,
        TabularDataSpecProtocol,
    ],
    Generic[GenerationParamsTCov],
):
    @classmethod
    @abstractmethod
    def _get_period(
        cls,
        data_split: DataSplit,
    ) -> Optional[int]: ...

    @classmethod
    @abstractmethod
    def _get_artifact_plan(cls, data_split: DataSplit) -> Optional[Type[TableComparisonPlan]]: ...

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
