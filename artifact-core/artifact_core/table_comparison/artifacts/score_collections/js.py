from dataclasses import dataclass
from typing import Dict

import pandas as pd

from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.libs.data_spec.tabular.protocol import (
    TabularDataSpecProtocol,
)
from artifact_core.libs.implementation.js.js import JSDistanceCalculator
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonScoreCollection,
)
@dataclass(frozen=True)
class JSDistanceHyperparams(ArtifactHyperparams):
    n_bins_cts_histogram: int
    categorical_only: bool
class JSDistance(TableComparisonScoreCollection[JSDistanceHyperparams]):
    def __init__(
        self,
        data_spec: TabularDataSpecProtocol,
        hyperparams: JSDistanceHyperparams,
    ):
        self._data_spec = data_spec
        self._hyperparams = hyperparams

    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, float]:
        dict_js = JSDistanceCalculator.compute_dict_js(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            ls_cts_features=self._data_spec.ls_cts_features,
            ls_cat_features=self._data_spec.ls_cat_features,
            cat_unique_map=self._data_spec.categorical_unique_map,
            n_bins_cts_histogram=self._hyperparams.n_bins_cts_histogram,
            categorical_only=self._hyperparams.categorical_only,
        )
        return dict_js
