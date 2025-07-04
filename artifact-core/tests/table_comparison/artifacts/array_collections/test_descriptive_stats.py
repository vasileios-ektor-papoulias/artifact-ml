from typing import Type
from unittest.mock import ANY

import numpy as np
import pandas as pd
import pytest
from artifact_core.base.artifact_dependencies import NO_ARTIFACT_HYPERPARAMS
from artifact_core.libs.implementation.tabular.descriptive_statistics.calculator import (
    DescriptiveStatistic,
    DescriptiveStatisticsCalculator,
)
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison.artifacts.array_collections.descriptive_stats import (
    FirstQuartileJuxtaposition,
    MaxJuxtaposition,
    MeanJuxtaposition,
    MedianJuxtaposition,
    MinJuxtaposition,
    STDJuxtaposition,
    ThirdQuartileJuxtaposition,
    VarianceJuxtaposition,
)
from artifact_core.table_comparison.artifacts.base import (
    DatasetComparisonArtifactResources,
    TableComparisonArrayCollection,
)
from pytest_mock import MockerFixture


@pytest.mark.parametrize(
    "artifact_class, statistic",
    [
        (MeanJuxtaposition, DescriptiveStatistic.MEAN),
        (STDJuxtaposition, DescriptiveStatistic.STD),
        (VarianceJuxtaposition, DescriptiveStatistic.VARIANCE),
        (MedianJuxtaposition, DescriptiveStatistic.MEDIAN),
        (FirstQuartileJuxtaposition, DescriptiveStatistic.Q1),
        (ThirdQuartileJuxtaposition, DescriptiveStatistic.Q3),
        (MinJuxtaposition, DescriptiveStatistic.MIN),
        (MaxJuxtaposition, DescriptiveStatistic.MAX),
    ],
)
def test_compute(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    artifact_class: Type[TableComparisonArrayCollection],
    statistic: DescriptiveStatistic,
):
    fake_result = {"f1": np.array([0.1, 0.2]), "f2": np.array([0.3, 0.4])}
    patch_compute = mocker.patch.object(
        target=DescriptiveStatisticsCalculator,
        attribute="compute_juxtaposition",
        return_value=fake_result,
    )
    artifact = artifact_class(resource_spec=resource_spec, hyperparams=NO_ARTIFACT_HYPERPARAMS)
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )
    result = artifact.compute(resources=resources)
    patch_compute.assert_called_once_with(
        df_real=ANY,
        df_synthetic=ANY,
        ls_cts_features=resource_spec.ls_cts_features,
        stat=statistic,
    )
    _, kwargs = patch_compute.call_args
    passed_real = kwargs["df_real"]
    passed_synth = kwargs["df_synthetic"]
    pd.testing.assert_frame_equal(passed_real, df_real)
    pd.testing.assert_frame_equal(passed_synth, df_synthetic)
    assert result == fake_result
