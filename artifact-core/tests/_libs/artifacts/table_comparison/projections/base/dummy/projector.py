from dataclasses import dataclass
from typing import List, Optional, Type, TypeVar

import numpy as np
import pandas as pd
from artifact_core._base.typing.artifact_result import Array
from artifact_core._libs.artifacts.table_comparison.projections.base.plotter import (
    ProjectionPlotter,
    ProjectionPlotterConfig,
)
from artifact_core._libs.artifacts.table_comparison.projections.base.projector import (
    ProjectorBase,
    ProjectorHyperparams,
)

dummyProjectorT = TypeVar("dummyProjectorT", bound="DummyProjector")


@dataclass(frozen=True)
class DummyProjectorHyperparams(ProjectorHyperparams):
    projection_type: str = "random"


class DummyProjector(ProjectorBase[DummyProjectorHyperparams]):
    @classmethod
    def build(
        cls: Type[dummyProjectorT],
        ls_cat_features: List[str],
        ls_cts_features: List[str],
        projector_config: Optional[DummyProjectorHyperparams] = None,
        plotter_config: Optional[ProjectionPlotterConfig] = None,
    ) -> dummyProjectorT:
        if projector_config is None:
            projector_config = DummyProjectorHyperparams()
        if plotter_config is None:
            plotter_config = ProjectionPlotterConfig()
        plotter = ProjectionPlotter(config=plotter_config)
        return cls(
            cat_features=ls_cat_features,
            cts_features=ls_cts_features,
            hyperparams=projector_config,
            plotter=plotter,
        )

    @classmethod
    def _get_projection_name(cls) -> str:
        return "dummy_projection"

    def _project(self, dataset_preprocessed: pd.DataFrame) -> Optional[Array]:
        if self._hyperparams.projection_type == "random":
            n_samples = len(dataset_preprocessed)
            n_cols = len(dataset_preprocessed.columns)
            return np.random.rand(n_samples, n_cols)
        else:
            raise ValueError(f"Unknown projection type: {self._hyperparams.projection_type}")
