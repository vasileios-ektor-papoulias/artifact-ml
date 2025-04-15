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
