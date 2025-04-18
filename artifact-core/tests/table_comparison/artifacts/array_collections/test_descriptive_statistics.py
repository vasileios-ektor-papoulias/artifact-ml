from types import SimpleNamespace
from typing import Callable, List, Type, cast
from unittest.mock import ANY

import numpy as np
import pandas as pd
import pytest
from artifact_core.libs.data_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.libs.implementation.descriptive_statistics.calculator import (
    DescriptiveStatistic,
    DescriptiveStatisticsCalculator,
)
from artifact_core.table_comparison.artifacts.array_collections.descriptive_statistics import (
    ContinuousFeatureFirstQuartilesJuxtaposition,
    ContinuousFeatureMaximaJuxtaposition,
    ContinuousFeatureMeansJuxtaposition,
    ContinuousFeatureMediansJuxtaposition,
    ContinuousFeatureMinimaJuxtaposition,
    ContinuousFeatureSTDsJuxtaposition,
    ContinuousFeatureThirdQuartilesJuxtaposition,
    ContinuousFeatureVariancesJuxtaposition,
)
from artifact_core.table_comparison.artifacts.base import (
    DatasetComparisonArtifactResources,
    TableComparisonArrayCollection,
)
from pytest_mock import MockerFixture


@pytest.fixture
def tabular_spec_factory() -> Callable[[List[str], List[str]], TabularDataSpecProtocol]:
    def _factory(
        ls_cts_features: List[str],
        ls_cat_feaetures: List[str],
    ) -> TabularDataSpecProtocol:
        ls_features = ls_cts_features + ls_cat_feaetures
        spec = SimpleNamespace(
            ls_features=ls_features,
            n_features=len(ls_features),
            ls_cts_features=ls_cts_features,
            n_cts_features=len(ls_cts_features),
            dict_cts_dtypes={},
            ls_cat_features=ls_cat_feaetures,
            n_cat_features=len(ls_cat_feaetures),
            dict_cat_dtypes={},
            cat_unique_map={},
            cat_unique_count_map={},
        )
        return cast(TabularDataSpecProtocol, spec)

    return _factory


@pytest.fixture
def df_real() -> pd.DataFrame:
    return pd.DataFrame({"c1": [1, 2], "c2": [3, 4]})


@pytest.fixture
def df_synth() -> pd.DataFrame:
    return pd.DataFrame({"c1": [5, 6], "c2": [7, 8]})


@pytest.mark.parametrize(
    "artifact_class, statistic",
    [
        (ContinuousFeatureMeansJuxtaposition, DescriptiveStatistic.MEAN),
        (ContinuousFeatureSTDsJuxtaposition, DescriptiveStatistic.STD),
        (ContinuousFeatureVariancesJuxtaposition, DescriptiveStatistic.VARIANCE),
        (ContinuousFeatureMediansJuxtaposition, DescriptiveStatistic.MEDIAN),
        (ContinuousFeatureFirstQuartilesJuxtaposition, DescriptiveStatistic.Q1),
        (ContinuousFeatureThirdQuartilesJuxtaposition, DescriptiveStatistic.Q3),
        (ContinuousFeatureMinimaJuxtaposition, DescriptiveStatistic.MIN),
        (ContinuousFeatureMaximaJuxtaposition, DescriptiveStatistic.MAX),
    ],
)
def test_compute(
    mocker: MockerFixture,
    tabular_spec_factory: Callable[[List[str], List[str]], TabularDataSpecProtocol],
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    artifact_class: Type[TableComparisonArrayCollection],
    statistic: DescriptiveStatistic,
):
    data_spec = tabular_spec_factory(["c1", "c2"], [])
    fake_result = {"f1": np.array([0.1, 0.2]), "f2": np.array([0.3, 0.4])}
    mock_compute = mocker.patch.object(
        DescriptiveStatisticsCalculator,
        "compute_juxtaposition",
        return_value=fake_result,
    )
    artifact = artifact_class(
        data_spec=data_spec,
    )
    resources = DatasetComparisonArtifactResources(dataset_real=df_real, dataset_synthetic=df_synth)
    result = artifact(resources=resources)
    mock_compute.assert_called_once_with(
        df_real=ANY,
        df_synthetic=ANY,
        ls_cts_features=data_spec.ls_cts_features,
        stat=statistic,
    )
    _, kwargs = mock_compute.call_args
    passed_real = kwargs["df_real"]
    passed_synth = kwargs["df_synthetic"]
    pd.testing.assert_frame_equal(passed_real, df_real)
    pd.testing.assert_frame_equal(passed_synth, df_synth)
    assert result == fake_result
