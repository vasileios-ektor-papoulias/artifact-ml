import pandas as pd

from artifact_core._base.core.hyperparams import NoArtifactHyperparams
from artifact_core._base.typing.artifact_result import ArrayCollection
from artifact_core._libs.artifacts.table_comparison.descriptive_stats.calculator import (
    TableStatsCalculator,
)
from artifact_core._libs.tools.calculators.descriptive_stats_calculator import DescriptiveStatistic
from artifact_core.table_comparison._artifacts.base import TableComparisonArrayCollection
from artifact_core.table_comparison._registries.array_collections import (
    TableComparisonArrayCollectionRegistry,
)
from artifact_core.table_comparison._types.array_collections import (
    TableComparisonArrayCollectionType,
)


@TableComparisonArrayCollectionRegistry.register_artifact(
    TableComparisonArrayCollectionType.MEAN_JUXTAPOSITION
)
class MeanJuxtapositionArrays(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> ArrayCollection:
        result = TableStatsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            cts_features=self._resource_spec.cts_features,
            stat=DescriptiveStatistic.MEAN,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(
    TableComparisonArrayCollectionType.STD_JUXTAPOSITION
)
class STDJuxtapositionArrays(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> ArrayCollection:
        result = TableStatsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            cts_features=self._resource_spec.cts_features,
            stat=DescriptiveStatistic.STD,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(
    TableComparisonArrayCollectionType.VARIANCE_JUXTAPOSITION
)
class VarianceJuxtapositionArrays(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> ArrayCollection:
        result = TableStatsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            cts_features=self._resource_spec.cts_features,
            stat=DescriptiveStatistic.VARIANCE,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(
    TableComparisonArrayCollectionType.MEDIAN_JUXTAPOSITION
)
class MedianJuxtapositionArrays(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> ArrayCollection:
        result = TableStatsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            cts_features=self._resource_spec.cts_features,
            stat=DescriptiveStatistic.MEDIAN,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(
    TableComparisonArrayCollectionType.FIRST_QUARTILE_JUXTAPOSITION
)
class FirstQuartileJuxtapositionArrays(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> ArrayCollection:
        result = TableStatsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            cts_features=self._resource_spec.cts_features,
            stat=DescriptiveStatistic.Q1,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(
    TableComparisonArrayCollectionType.THIRD_QUARTILE_JUXTAPOSITION
)
class ThirdQuartileJuxtapositionArrays(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> ArrayCollection:
        result = TableStatsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            cts_features=self._resource_spec.cts_features,
            stat=DescriptiveStatistic.Q3,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(
    TableComparisonArrayCollectionType.MIN_JUXTAPOSITION
)
class MinJuxtapositionArrays(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> ArrayCollection:
        result = TableStatsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            cts_features=self._resource_spec.cts_features,
            stat=DescriptiveStatistic.MIN,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(
    TableComparisonArrayCollectionType.MAX_JUXTAPOSITION
)
class MaxJuxtapositionArrays(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> ArrayCollection:
        result = TableStatsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            cts_features=self._resource_spec.cts_features,
            stat=DescriptiveStatistic.MAX,
        )
        return result
