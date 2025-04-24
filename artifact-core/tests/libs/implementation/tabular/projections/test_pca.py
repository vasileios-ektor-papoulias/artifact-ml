from typing import List, Tuple

import pandas as pd
import pytest
from artifact_core.libs.implementation.tabular.projections.pca import PCAHyperparams, PCAProjector
from pytest_mock import MockerFixture
from sklearn.decomposition import PCA


@pytest.mark.parametrize(
    "dataset_dispatcher, projector_config",
    [
        ("dataset_small", PCAHyperparams(use_categorical=True)),
        ("dataset_small", PCAHyperparams(use_categorical=False)),
        ("dataset_large", PCAHyperparams(use_categorical=True)),
        ("dataset_large", PCAHyperparams(use_categorical=False)),
        ("dataset_mixed", PCAHyperparams(use_categorical=True)),
        ("dataset_mixed", PCAHyperparams(use_categorical=False)),
    ],
    indirect=["dataset_dispatcher"],
)
def test_project(
    mocker: MockerFixture,
    dataset_dispatcher: Tuple[pd.DataFrame, List[str], List[str]],
    projector_config: PCAHyperparams,
):
    df, ls_cat_features, ls_cts_features = dataset_dispatcher
    projector = PCAProjector.build(
        ls_cat_features=ls_cat_features,
        ls_cts_features=ls_cts_features,
        projector_config=projector_config,
    )
    mock_projection = "mock_projection"
    patcher_pca = mocker.patch.object(
        target=PCA, attribute="fit_transform", return_value=mock_projection
    )
    result_projection = projector.project(dataset=df)
    patcher_pca.assert_called_once()
    _, kwargs = patcher_pca.call_args
    passed_dataset = kwargs["X"]
    expected_passed_dataset = (
        pd.get_dummies(df, columns=ls_cat_features)
        if projector_config.use_categorical
        else df[ls_cts_features]
    )
    pd.testing.assert_frame_equal(passed_dataset, expected_passed_dataset)
    assert result_projection == mock_projection
