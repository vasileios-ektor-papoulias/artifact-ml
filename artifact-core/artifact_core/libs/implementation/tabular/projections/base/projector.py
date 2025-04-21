from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from artifact_core.libs.implementation.tabular.projections.base.plotter import (
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
        self._validate_data_spec(ls_cat_features=ls_cat_features, ls_cts_features=ls_cts_features)
        self._ls_cat_features = ls_cat_features
        self._ls_cts_features = ls_cts_features
        self._hyperparams = hyperparams
        self._plotter = plotter
        self._projection_name = self._get_projection_name()

    @property
    def projection_name(self) -> str:
        return self._projection_name

    def project(self, dataset: pd.DataFrame) -> Optional[np.ndarray]:
        dataset_preprocessed = self._preprocess(dataset=dataset)
        projection = self._project(dataset_preprocessed=dataset_preprocessed)
        return projection

    @classmethod
    @abstractmethod
    def _get_projection_name(cls) -> str: ...

    @abstractmethod
    def _project(self, dataset_preprocessed: pd.DataFrame) -> Optional[np.ndarray]: ...

    def produce_projection_plot(self, dataset: pd.DataFrame) -> Figure:
        dataset_projection = self.project(dataset=dataset)
        fig_projection = self._plotter.produce_projection_plot(
            dataset_projection_2d=dataset_projection,
            projection_name=self.projection_name,
        )
        return fig_projection

    def produce_projection_comparison_plot(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        dataset_projection_real = self.project(dataset=dataset_real)
        dataset_projection_synthetic = self.project(dataset=dataset_synthetic)
        fig_projection_comparison = self._plotter.produce_projection_comparison_plot(
            dataset_projection_2d_real=dataset_projection_real,
            dataset_projection_2d_synthetic=dataset_projection_synthetic,
            projection_name=self.projection_name,
        )
        return fig_projection_comparison

    def _preprocess(self, dataset: pd.DataFrame) -> pd.DataFrame:
        cts_data = dataset[self._ls_cts_features]
        if self._hyperparams.use_categorical and self._ls_cat_features:
            cat_data = dataset[self._ls_cat_features]
            cat_data_encoded = pd.get_dummies(cat_data, drop_first=False)
            dataset_preprocessed = pd.concat([cts_data, cat_data_encoded], axis=1)
        else:
            dataset_preprocessed = cts_data
        return dataset_preprocessed

    @staticmethod
    def _validate_data_spec(ls_cat_features: List[str], ls_cts_features: List[str]):
        if not ls_cat_features and not ls_cts_features:
            raise ValueError("Both categorical and continuous feature lists are empty.")
        overlap = set(ls_cat_features).intersection(ls_cts_features)
        if overlap:
            raise ValueError(f"Categorical and continuous features overlap: {overlap}")
