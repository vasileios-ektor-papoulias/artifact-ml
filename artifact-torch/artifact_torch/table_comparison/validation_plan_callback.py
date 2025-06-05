from typing import Any, Generic, Optional, TypeVar

import pandas as pd
from artifact_core.table_comparison.artifacts.base import TableComparisonArtifactResources
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.table_comparison.validation_plan import TableComparisonPlan

from artifact_torch.base.components.callbacks.validation_plan import (
    ValidationPlanCallback,
)
from artifact_torch.core.model.generative import GenerationParams
from artifact_torch.libs.exports.table import TableExporter
from artifact_torch.table_comparison.model import TabularGenerativeModel

GenerationParamsT = TypeVar("GenerationParamsT", bound=GenerationParams)


class TableComparisonPlanCallback(
    ValidationPlanCallback[
        TabularGenerativeModel[Any, Any, GenerationParamsT],
        TableComparisonPlan,
        TableComparisonArtifactResources,
    ],
    Generic[GenerationParamsT],
):
    _resource_export_prefix = "synthetic"

    def __init__(
        self,
        period: int,
        validation_plan: TableComparisonPlan,
        df_real: pd.DataFrame,
        generation_params: GenerationParamsT,
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(period=period, validation_plan=validation_plan)
        self._df_real = df_real
        self._generation_params = generation_params
        self._tracking_client = tracking_client

    def _generate_artifact_resources(
        self,
        model: TabularGenerativeModel[Any, Any, GenerationParamsT],
    ) -> TableComparisonArtifactResources:
        df_synthetic = model.generate(params=self._generation_params)
        resources = TableComparisonArtifactResources(
            dataset_real=self._df_real, dataset_synthetic=df_synthetic
        )
        return resources

    def _export_artifact_resources(
        self,
        artifact_resources: TableComparisonArtifactResources,
        step: int,
    ):
        if self._tracking_client is not None:
            TableExporter.export(
                data=artifact_resources.dataset_synthetic,
                tracking_client=self._tracking_client,
                prefix=self._resource_export_prefix,
                step=step,
            )
