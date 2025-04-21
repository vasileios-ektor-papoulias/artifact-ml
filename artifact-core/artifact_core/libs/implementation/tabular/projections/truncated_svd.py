from dataclasses import dataclass
from typing import List, Optional, Type, TypeVar

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import NotFittedError

from artifact_core.libs.implementation.tabular.projections.base.plotter import (
    ProjectionPlotter,
    ProjectionPlotterConfig,
)
from artifact_core.libs.implementation.tabular.projections.base.projector import (
    ProjectorBase,
    ProjectorHyperparams,
)

truncatedSVDProjectorType = TypeVar("truncatedSVDProjectorType", bound="TruncatedSVDProjector")


@dataclass(frozen=True)
class TruncatedSVDHyperparams(ProjectorHyperparams):
    n_components: int = 2


class TruncatedSVDProjector(ProjectorBase[TruncatedSVDHyperparams]):
    @classmethod
    def build(
        cls: Type[truncatedSVDProjectorType],
        ls_cat_features: List[str],
        ls_cts_features: List[str],
        projector_config: Optional[TruncatedSVDHyperparams] = None,
        plotter_config: Optional[ProjectionPlotterConfig] = None,
    ) -> truncatedSVDProjectorType:
        if projector_config is None:
            projector_config = TruncatedSVDHyperparams()
        if plotter_config is None:
            plotter_config = ProjectionPlotterConfig()

        plotter = ProjectionPlotter(config=plotter_config)
        return cls(
            ls_cat_features=ls_cat_features,
            ls_cts_features=ls_cts_features,
            hyperparams=projector_config,
            plotter=plotter,
        )

    def _project(self, dataset_preprocessed: pd.DataFrame) -> np.ndarray | None:
        try:
            tsvd_model = TruncatedSVD(n_components=self._hyperparams.n_components)
            return tsvd_model.fit_transform(X=dataset_preprocessed)
        except (ValueError, LinAlgError, NotFittedError):
            return None

    @classmethod
    def _get_projection_name(cls) -> str:
        return "TruncatedSVD"
