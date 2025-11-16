import pandas as pd
import pytest
from artifact_core._domains.dataset_comparison.artifact import DatasetComparisonArtifactResources
from artifact_experiment.core.dataset_comparison.callback_resources import (
    DatasetComparisonCallbackResources,
)
from artifact_experiment.table_comparison._callback_resources import (
    TableComparisonCallbackResources,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dataset_real_dispatcher, dataset_synthetic_dispatcher",
    [
        ("df_1", "df_2"),
        ("df_3", "df_4"),
        ("df_5", "df_5"),
    ],
    indirect=["dataset_real_dispatcher", "dataset_synthetic_dispatcher"],
)
def test_build(dataset_real_dispatcher: pd.DataFrame, dataset_synthetic_dispatcher: pd.DataFrame):
    dataset_real = dataset_real_dispatcher
    dataset_synthetic = dataset_synthetic_dispatcher
    resources = TableComparisonCallbackResources.build(
        dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
    )

    assert isinstance(resources, DatasetComparisonCallbackResources)
    assert isinstance(resources.artifact_resources, DatasetComparisonArtifactResources)
    assert isinstance(resources.artifact_resources.dataset_real, pd.DataFrame)
    assert isinstance(resources.artifact_resources.dataset_synthetic, pd.DataFrame)
    pd.testing.assert_frame_equal(resources.artifact_resources.dataset_real, dataset_real)
    pd.testing.assert_frame_equal(resources.artifact_resources.dataset_synthetic, dataset_synthetic)
