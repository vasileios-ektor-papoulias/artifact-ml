from typing import Dict

import numpy as np
import pandas as pd

from artifact_core.base.artifact_dependencies import NoArtifactHyperparams
from artifact_core.libs.implementation.descriptive_statistics.calculator import (
    DescriptiveStatistic,
    DescriptiveStatisticsCalculator,
)
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonArrayCollection,
)
from artifact_core.table_comparison.registries.array_collections.registry import (
    TableComparisonArrayCollectionRegistry,
    TableComparisonArrayCollectionType,
)


@TableComparisonArrayCollectionRegistry.register_artifact(TableComparisonArrayCollectionType.MEANS)
class ContinuousFeatureMeansJuxtaposition(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        result = DescriptiveStatisticsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            ls_cts_features=self._data_spec.ls_cts_features,
            stat=DescriptiveStatistic.MEAN,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(TableComparisonArrayCollectionType.STDS)
class ContinuousFeatureSTDsJuxtaposition(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        result = DescriptiveStatisticsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            ls_cts_features=self._data_spec.ls_cts_features,
            stat=DescriptiveStatistic.STD,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(
    TableComparisonArrayCollectionType.VARIANCES
)
class ContinuousFeatureVariancesJuxtaposition(
    TableComparisonArrayCollection[NoArtifactHyperparams]
):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        result = DescriptiveStatisticsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            ls_cts_features=self._data_spec.ls_cts_features,
            stat=DescriptiveStatistic.VARIANCE,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(
    TableComparisonArrayCollectionType.MEDIANS
)
class ContinuousFeatureMediansJuxtaposition(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        result = DescriptiveStatisticsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            ls_cts_features=self._data_spec.ls_cts_features,
            stat=DescriptiveStatistic.MEDIAN,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(
    TableComparisonArrayCollectionType.FIRST_QUARTILES
)
class ContinuousFeatureFirstQuartilesJuxtaposition(
    TableComparisonArrayCollection[NoArtifactHyperparams]
):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        result = DescriptiveStatisticsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            ls_cts_features=self._data_spec.ls_cts_features,
            stat=DescriptiveStatistic.Q1,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(
    TableComparisonArrayCollectionType.THIRD_QUARTILES
)
class ContinuousFeatureThirdQuartilesJuxtaposition(
    TableComparisonArrayCollection[NoArtifactHyperparams]
):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        result = DescriptiveStatisticsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            ls_cts_features=self._data_spec.ls_cts_features,
            stat=DescriptiveStatistic.Q3,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(TableComparisonArrayCollectionType.MINIMA)
class ContinuousFeatureMinimaJuxtaposition(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        result = DescriptiveStatisticsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            ls_cts_features=self._data_spec.ls_cts_features,
            stat=DescriptiveStatistic.MIN,
        )
        return result


@TableComparisonArrayCollectionRegistry.register_artifact(TableComparisonArrayCollectionType.MAXIMA)
class ContinuousFeatureMaximaJuxtaposition(TableComparisonArrayCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        result = DescriptiveStatisticsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            ls_cts_features=self._data_spec.ls_cts_features,
            stat=DescriptiveStatistic.MAX,
        )
        return result
