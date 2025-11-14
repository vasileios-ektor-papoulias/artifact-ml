from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Type, TypeVar

import pandas as pd
from numpy.linalg import LinAlgError
from sklearn.exceptions import NotFittedError
from sklearn.manifold import TSNE

from artifact_core._base.typing.artifact_result import Array
from artifact_core._libs.artifacts.table_comparison.projections.base.plotter import (
    ProjectionPlotter,
    ProjectionPlotterConfig,
)
from artifact_core._libs.artifacts.table_comparison.projections.base.projector import (
    ProjectorBase,
    ProjectorHyperparams,
)

TSNEProjectorT = TypeVar("TSNEProjectorT", bound="TSNEProjector")


@dataclass(frozen=True)
class TSNEHyperparams(ProjectorHyperparams):
    perplexity: float = 30.0
    learning_rate: float | Literal["auto"] = "auto"
    max_iter: int = 1000


class TSNEProjector(ProjectorBase[TSNEHyperparams]):
    _n_components = 2

    @classmethod
    def build(
        cls: Type[TSNEProjectorT],
        cat_features: Sequence[str],
        cts_features: Sequence[str],
        projector_config: Optional[TSNEHyperparams] = None,
        plotter_config: Optional[ProjectionPlotterConfig] = None,
    ) -> TSNEProjectorT:
        if projector_config is None:
            projector_config = TSNEHyperparams()
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

    def _project(self, dataset_preprocessed: pd.DataFrame) -> Array | None:
        try:
            tsne_model = TSNE(
                n_components=self._n_components,
                perplexity=self._hyperparams.perplexity,
                learning_rate=self._hyperparams.learning_rate,
                max_iter=self._hyperparams.max_iter,
            )
            return tsne_model.fit_transform(X=dataset_preprocessed)
        except (ValueError, LinAlgError, NotFittedError):
            return None

    @classmethod
    def _get_projection_name(cls) -> str:
        return "t-SNE"
