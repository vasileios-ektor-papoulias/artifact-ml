from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from artifact_core.libs.implementation.projections.base.plotter import (
    ProjectionPlotter,
)

projectorHyperparamsT = TypeVar("projectorHyperparamsT", bound="ProjectorHyperparams")


@dataclass(frozen=True)
class ProjectorHyperparams:
    use_categorical: bool = True


class ProjectorBase(ABC, Generic[projectorHyperparamsT]):
    def __init__(
        self,
        ls_cat_features: List[str],
        ls_cts_features: List[str],
        hyperparams: projectorHyperparamsT,
        plotter: ProjectionPlotter,
    ):
        self._ls_cat_features = ls_cat_features
        self._ls_cts_features = ls_cts_features
        self._hyperparams = hyperparams
        self._plotter = plotter
        self._projection_name = self._get_projection_name()

    @property
    def projection_name(self) -> str:
        return self._projection_name

    @classmethod
    @abstractmethod
    def _get_projection_name(cls) -> str: ...

    @abstractmethod
    def project(self, dataset: pd.DataFrame) -> Optional[np.ndarray]: ...

    def compute_projection_plot(self, dataset: pd.DataFrame) -> Figure:
        dataset_projection = self.project(dataset=dataset)
        fig_projection = self._plotter.get_projection_plot(
            dataset_projection_2d=dataset_projection,
            method_name=self.projection_name,
        )
        return fig_projection

    def compute_projection_comparison_plot(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        dataset_projection_real = self.project(dataset=dataset_real)
        dataset_projection_synthetic = self.project(dataset=dataset_synthetic)
        fig_projection_comparison = self._plotter.get_projection_comparison_plot(
            dataset_projection_2d_real=dataset_projection_real,
            dataset_projection_2d_synthetic=dataset_projection_synthetic,
            method_name=self.projection_name,
        )
        return fig_projection_comparison

    def _prepare_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        cts_data = dataset[self._ls_cts_features]
        if self._hyperparams.use_categorical and self._ls_cat_features:
            cat_data = dataset[self._ls_cat_features]
            cat_data_encoded = pd.get_dummies(cat_data, drop_first=False)
            combined = pd.concat([cts_data, cat_data_encoded], axis=1)
        else:
            combined = cts_data
        return combined
