from typing import Type
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
def test_call(
    mocker: MockerFixture,
    data_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    artifact_class: Type[TableComparisonArrayCollection],
    statistic: DescriptiveStatistic,
):
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
