from typing import Dict

import numpy as np
import pandas as pd

from artifact_core.base.artifact_dependencies import NoArtifactHyperparams
from artifact_core.libs.implementation.tabular.descriptive_stats.calculator import (
    DescriptiveStatistic,
    DescriptiveStatsCalculator,
)
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonArrayCollection,
)
from artifact_core.table_comparison.registries.array_collections.registry import (
    TableComparisonArrayCollectionRegistry,
    TableComparisonArrayCollectionType,
)


@TableComparisonArrayCollectionRegistry.register_artifact(
    TableComparisonArrayCollectionType.MEAN_JUXTAPOSITION
)
class MeanJuxtapositionArrays(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        result = DescriptiveStatsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            ls_cts_features=self._resource_spec.ls_cts_features,
            stat=DescriptiveStatistic.MEAN,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(
    TableComparisonArrayCollectionType.STD_JUXTAPOSITION
)
class STDJuxtapositionArrays(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        result = DescriptiveStatsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            ls_cts_features=self._resource_spec.ls_cts_features,
            stat=DescriptiveStatistic.STD,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(
    TableComparisonArrayCollectionType.VARIANCE_JUXTAPOSITION
)
class VarianceJuxtapositionArrays(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        result = DescriptiveStatsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            ls_cts_features=self._resource_spec.ls_cts_features,
            stat=DescriptiveStatistic.VARIANCE,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(
    TableComparisonArrayCollectionType.MEDIAN_JUXTAPOSITION
)
class MedianJuxtapositionArrays(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        result = DescriptiveStatsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            ls_cts_features=self._resource_spec.ls_cts_features,
            stat=DescriptiveStatistic.MEDIAN,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(
    TableComparisonArrayCollectionType.FIRST_QUARTILE_JUXTAPOSITION
)
class FirstQuartileJuxtapositionArrays(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        result = DescriptiveStatsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            ls_cts_features=self._resource_spec.ls_cts_features,
            stat=DescriptiveStatistic.Q1,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(
    TableComparisonArrayCollectionType.THIRD_QUARTILE_JUXTAPOSITION
)
class ThirdQuartileJuxtapositionArrays(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        result = DescriptiveStatsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            ls_cts_features=self._resource_spec.ls_cts_features,
            stat=DescriptiveStatistic.Q3,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(
    TableComparisonArrayCollectionType.MIN_JUXTAPOSITION
)
class MinJuxtapositionArrays(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        result = DescriptiveStatsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            ls_cts_features=self._resource_spec.ls_cts_features,
            stat=DescriptiveStatistic.MIN,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(
    TableComparisonArrayCollectionType.MAX_JUXTAPOSITION
)
class MaxJuxtapositionArrays(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        result = DescriptiveStatsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            ls_cts_features=self._resource_spec.ls_cts_features,
            stat=DescriptiveStatistic.MAX,
        )
        return result
