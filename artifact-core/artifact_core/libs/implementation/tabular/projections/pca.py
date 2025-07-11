from dataclasses import dataclass
from typing import List, Optional, Type, TypeVar

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

from artifact_core.libs.implementation.tabular.projections.base.plotter import (
    ProjectionPlotter,
    ProjectionPlotterConfig,
)
from artifact_core.libs.implementation.tabular.projections.base.projector import (
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
        ls_cat_features: List[str],
        ls_cts_features: List[str],
        projector_config: Optional[PCAHyperparams] = None,
        plotter_config: Optional[ProjectionPlotterConfig] = None,
    ) -> PCAProjectorT:
        if projector_config is None:
            projector_config = PCAHyperparams()
        if plotter_config is None:
            plotter_config = ProjectionPlotterConfig()
        plotter = ProjectionPlotter(config=plotter_config)
        projector = cls(
            ls_cat_features=ls_cat_features,
            ls_cts_features=ls_cts_features,
            hyperparams=projector_config,
            plotter=plotter,
        )
        return projector

    def _project(self, dataset_preprocessed: pd.DataFrame) -> Optional[np.ndarray]:
        try:
            pca_model = PCA(n_components=self._n_components)
            return pca_model.fit_transform(X=dataset_preprocessed)
        except (ValueError, np.linalg.LinAlgError, NotFittedError):
            return None

    @classmethod
    def _get_projection_name(cls) -> str:
        return "PCA"
