from dataclasses import dataclass
from typing import List, Literal, Optional, Type, TypeVar

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from sklearn.exceptions import NotFittedError
from sklearn.manifold import TSNE

from artifact_core.libs.implementation.tabular.projections.base.plotter import (
    ProjectionPlotter,
    ProjectionPlotterConfig,
)
from artifact_core.libs.implementation.tabular.projections.base.projector import (
    ProjectorBase,
    ProjectorHyperparams,
)

tsneProjectorType = TypeVar("tsneProjectorType", bound="TSNEProjector")


@dataclass(frozen=True)
class TSNEHyperparams(ProjectorHyperparams):
    perplexity: float = 30.0
    learning_rate: float | Literal["auto"] = "auto"
    n_iter: int = 1000


class TSNEProjector(ProjectorBase[TSNEHyperparams]):
    _n_components = 2

    @classmethod
    def build(
        cls: Type[tsneProjectorType],
        ls_cat_features: List[str],
        ls_cts_features: List[str],
        projector_config: Optional[TSNEHyperparams] = None,
        plotter_config: Optional[ProjectionPlotterConfig] = None,
    ) -> tsneProjectorType:
        if projector_config is None:
            projector_config = TSNEHyperparams()
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

    def _project(self, dataset_preprocessed: pd.DataFrame) -> np.ndarray | None:
        try:
            tsne_model = TSNE(
                n_components=self._n_components,
                perplexity=self._hyperparams.perplexity,
                learning_rate=self._hyperparams.learning_rate,
                n_iter=self._hyperparams.n_iter,
            )
            return tsne_model.fit_transform(X=dataset_preprocessed)
        except (ValueError, LinAlgError, NotFittedError):
            return None

    @classmethod
    def _get_projection_name(cls) -> str:
        return "t-SNE"
