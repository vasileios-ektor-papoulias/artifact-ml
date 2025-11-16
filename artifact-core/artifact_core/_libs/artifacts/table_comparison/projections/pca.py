from dataclasses import dataclass
from typing import Optional, Sequence, Type, TypeVar

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

from artifact_core._base.typing.artifact_result import Array
from artifact_core._libs.artifacts.table_comparison.projections.base.plotter import (
    ProjectionPlotter,
    ProjectionPlotterConfig,
)
from artifact_core._libs.artifacts.table_comparison.projections.base.projector import (
    ProjectorBase,
    ProjectorHyperparams,
)

PCAProjectorT = TypeVar("PCAProjectorT", bound="PCAProjector")


@dataclass(frozen=True)
class PCAHyperparams(ProjectorHyperparams):
    pass


class PCAProjector(ProjectorBase[PCAHyperparams]):
    _n_components = 2

    @classmethod
    def build(
        cls: Type[PCAProjectorT],
        cat_features: Sequence[str],
        cts_features: Sequence[str],
        projector_config: Optional[PCAHyperparams] = None,
        plotter_config: Optional[ProjectionPlotterConfig] = None,
    ) -> PCAProjectorT:
        if projector_config is None:
            projector_config = PCAHyperparams()
        if plotter_config is None:
            plotter_config = ProjectionPlotterConfig()
        plotter = ProjectionPlotter(config=plotter_config)
        projector = cls(
            cat_features=cat_features,
            cts_features=cts_features,
            hyperparams=projector_config,
            plotter=plotter,
        )
        return projector

    def _project(self, dataset_preprocessed: pd.DataFrame) -> Optional[Array]:
        try:
            pca_model = PCA(n_components=self._n_components)
            return pca_model.fit_transform(X=dataset_preprocessed)
        except (ValueError, np.linalg.LinAlgError, NotFittedError):
            return None

    @classmethod
    def _get_projection_name(cls) -> str:
        return "PCA"
