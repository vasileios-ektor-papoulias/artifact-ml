from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Mapping, Optional, Type, TypeVar

import pandas as pd
from artifact_core._libs.resources_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison._artifacts.base import TableComparisonArtifactResources
from artifact_experiment import DataSplit
from artifact_experiment.base.tracking.background.tracking_queue import TrackingQueue
from artifact_experiment.table_comparison.plan import TableComparisonPlan

from artifact_torch.base.components.callbacks.export import ExportCallback
from artifact_torch.base.components.routines.artifact import (
    ArtifactRoutine,
    ArtifactRoutineData,
    ArtifactRoutineHyperparams,
)
from artifact_torch.core.model.generative import GenerationParams
from artifact_torch.libs.components.callbacks.export.table import (
    TableExportCallback,
    TableExportCallbackResources,
)
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
        TableComparisonRoutineHyperparams[GenerationParamsTCov],
        TableComparisonRoutineData,
        TableComparisonArtifactResources,
        TabularDataSpecProtocol,
        pd.DataFrame,
    ],
    Generic[GenerationParamsTCov],
):
    _resource_export_trigger_identifier = "SYNTHETIC"

    @classmethod
    @abstractmethod
    def _get_period(
        cls,
        data_split: DataSplit,
    ) -> Optional[int]: ...

    @classmethod
    @abstractmethod
    def _get_generation_params(cls) -> GenerationParamsTCov: ...

    @classmethod
    @abstractmethod
    def _get_artifact_plan(cls, data_split: DataSplit) -> Optional[Type[TableComparisonPlan]]: ...

    @classmethod
    def _get_hyperparams(cls) -> TableComparisonRoutineHyperparams[GenerationParamsTCov]:
        generation_params = cls._get_generation_params()
        hyperparams = TableComparisonRoutineHyperparams[GenerationParamsTCov](
            generation_params=generation_params
        )
        return hyperparams

    @classmethod
    def _get_export_callback(
        cls, tracking_queue: Optional[TrackingQueue]
    ) -> Optional[ExportCallback[pd.DataFrame]]:
        if tracking_queue is not None:
            return TableExportCallback(period=1, writer=tracking_queue.file_writer)

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
        export_callback: ExportCallback[pd.DataFrame],
        n_epochs_elapsed: int,
        data_split: DataSplit,
    ):
        _ = data_split
        export_callback_resources = TableExportCallbackResources(
            step=n_epochs_elapsed,
            export_data=artifact_resources.dataset_synthetic,
            trigger=cls._resource_export_trigger_identifier,
        )
        export_callback.execute(resources=export_callback_resources)
