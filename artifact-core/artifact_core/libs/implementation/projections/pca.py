from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

from artifact_core.libs.implementation.projections.base.plotter import (
    ProjectionPlotter,
    ProjectionPlotterConfig,
)
from artifact_core.libs.implementation.projections.base.projector import (
    ProjectorBase,
    ProjectorHyperparams,
)


@dataclass(frozen=True)
class PCAHyperparams(ProjectorHyperparams):
    n_components: int = 2


class PCAProjector(ProjectorBase[PCAHyperparams]):
    @classmethod
    def build(
        cls,
        ls_cat_features: List[str],
        ls_cts_features: List[str],
        projector_config: Optional[PCAHyperparams] = None,
        plotter_config: Optional[ProjectionPlotterConfig] = None,
    ) -> "PCAProjector":
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

    def project(self, dataset: pd.DataFrame) -> np.ndarray | None:
        combined = self._prepare_data(dataset=dataset)
        try:
            pca_model = PCA(n_components=self._hyperparams.n_components)
            return pca_model.fit_transform(X=combined)
        except (ValueError, np.linalg.LinAlgError, NotFittedError):
            return None

    @classmethod
    def _get_projection_name(cls) -> str:
        return "PCA"
