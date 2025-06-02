from typing import Any, Generic, TypeVar

import pandas as pd
from artifact_core.table_comparison.artifacts.base import TableComparisonArtifactResources
from artifact_experiment.table_comparison.validation_plan import TableComparisonValidationPlan

from artifact_torch.base.components.callbacks.validation_plan import ValidationPlanCallback
from artifact_torch.core.model.generative import GenerationParams
from artifact_torch.table_comparison.model import TabularGenerativeModel

GenerationParamsT = TypeVar("GenerationParamsT", bound=GenerationParams)


class TableComparisonPlanCallback(
    ValidationPlanCallback[
        TabularGenerativeModel[Any, Any, GenerationParamsT],
        TableComparisonValidationPlan,
        TableComparisonArtifactResources,
    ],
    Generic[GenerationParamsT],
):
    def __init__(
        self,
        period: int,
        validation_plan: TableComparisonValidationPlan,
        df_real: pd.DataFrame,
        generation_params: GenerationParamsT,
    ):
        super().__init__(period=period, validation_plan=validation_plan)
        self._df_real = df_real
        self._generation_params = generation_params

    def _generate_artifact_resources(
        self,
        model: TabularGenerativeModel[Any, Any, GenerationParamsT],
    ) -> TableComparisonArtifactResources:
        df_synthetic = model.generate(params=self._generation_params)
        resources = TableComparisonArtifactResources(
            dataset_real=self._df_real, dataset_synthetic=df_synthetic
        )
        return resources
