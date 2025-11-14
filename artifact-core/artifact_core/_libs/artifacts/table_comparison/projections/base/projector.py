from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, Sequence, TypeVar

import pandas as pd
from matplotlib.figure import Figure

from artifact_core._base.typing.artifact_result import Array
from artifact_core._libs.artifacts.table_comparison.projections.base.plotter import (
    ProjectionPlotter,
)

ProjectorHyperparamsT = TypeVar("ProjectorHyperparamsT", bound="ProjectorHyperparams")


@dataclass(frozen=True)
class ProjectorHyperparams:
    use_categorical: bool = True


class ProjectorBase(ABC, Generic[ProjectorHyperparamsT]):
    def __init__(
        self,
        cat_features: Sequence[str],
        cts_features: Sequence[str],
        hyperparams: ProjectorHyperparamsT,
        plotter: ProjectionPlotter,
    ):
        self._validate_resource_spec(cat_features=cat_features, cts_features=cts_features)
        self._ls_cat_features = cat_features
        self._ls_cts_features = cts_features
        self._hyperparams = hyperparams
        self._plotter = plotter
        self._projection_name = self._get_projection_name()

    @property
    def projection_name(self) -> str:
        return self._projection_name

    def project(self, dataset: pd.DataFrame) -> Optional[Array]:
        dataset_preprocessed = self._preprocess(dataset=dataset)
        projection = self._project(dataset_preprocessed=dataset_preprocessed)
        return projection

    @classmethod
    @abstractmethod
    def _get_projection_name(cls) -> str: ...

    @abstractmethod
    def _project(self, dataset_preprocessed: pd.DataFrame) -> Optional[Array]: ...

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
    def _validate_resource_spec(cat_features: Sequence[str], cts_features: Sequence[str]):
        if not cat_features and not cts_features:
            raise ValueError("Both categorical and continuous feature lists are empty.")
        overlap = set(cat_features).intersection(cts_features)
        if overlap:
            raise ValueError(f"Categorical and continuous features overlap: {overlap}")
