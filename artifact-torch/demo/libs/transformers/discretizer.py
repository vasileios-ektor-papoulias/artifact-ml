from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class Discretizer:
    def __init__(self, n_bins: int = 5, ls_cts_features: Optional[List[str]] = None):
        self._n_bins = n_bins
        if ls_cts_features is None:
            ls_cts_features = []
        self._ls_cts_features = ls_cts_features
        self._bin_edges: Dict[str, np.ndarray] = {}
        self._is_fitted: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        n_bins: Optional[int] = None,
        ls_cts_features: Optional[List[str]] = None,
    ) -> "Discretizer":
        if ls_cts_features is not None:
            self._ls_cts_features = ls_cts_features
        if n_bins is not None:
            self._n_bins = n_bins
        self._bin_edges = self._fit(
            df=df, ls_cts_features=self._ls_cts_features, n_bins=self._n_bins
        )
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        return self._transform(
            df=df, continuous_features=self._ls_cts_features, bin_edges=self._bin_edges
        )

    def inverse_transform(self, df_binned: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        return self._inverse_transform(df_binned=df_binned, bin_edges=self._bin_edges)

    @classmethod
    def _fit(
        cls, df: pd.DataFrame, ls_cts_features: List[str], n_bins: int
    ) -> Dict[str, np.ndarray]:
        bin_edges: Dict[str, np.ndarray] = {}
        for feature in ls_cts_features:
            cls._validate_feature(df=df, feature=feature)
            _, edges = cls._bin_feature(series=df[feature], bins=n_bins)
            bin_edges[feature] = edges
        return bin_edges

    @classmethod
    def _transform(
        cls, df: pd.DataFrame, continuous_features: List[str], bin_edges: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        df_binned = df.copy()

        for feature in continuous_features:
            cls._validate_feature(df=df_binned, feature=feature)
            binned_series, _ = cls._bin_feature(
                series=df_binned[feature], bins=0, bin_edges=bin_edges[feature]
            )
            df_binned[feature] = binned_series

        return df_binned

    @classmethod
    def _inverse_transform(
        cls, df_binned: pd.DataFrame, bin_edges: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        df_reconstructed = df_binned.copy()

        for feature, edges in bin_edges.items():
            binned_col = feature
            if binned_col not in df_reconstructed.columns:
                raise ValueError(
                    f"Column '{binned_col}' not found in DataFrame for inverse transformation."
                )
            df_reconstructed[feature] = cls._sample_uniform(
                binned_series=df_reconstructed[binned_col], bin_edges=edges
            )

        return df_reconstructed

    @staticmethod
    def _bin_feature(
        series: pd.Series, bins: int, bin_edges: Optional[np.ndarray] = None
    ) -> Tuple[pd.Series, np.ndarray]:
        if bin_edges is None:
            binned, edges = pd.cut(
                x=series, bins=bins, labels=False, retbins=True, duplicates="drop"
            )
            return binned, edges
        else:
            binned = pd.cut(x=series, bins=bin_edges.tolist(), labels=False, duplicates="drop")
            return binned, bin_edges

    @classmethod
    def _sample_uniform(cls, binned_series: pd.Series, bin_edges: np.ndarray) -> pd.Series:
        return pd.Series(
            data=[
                cls._sample_single_uniform(idx=idx, bin_edges=bin_edges) for idx in binned_series
            ],
            index=binned_series.index,
        )

    @staticmethod
    def _sample_single_uniform(idx: Any, bin_edges: np.ndarray) -> float:
        if pd.isna(idx):
            return np.nan
        left = bin_edges[int(idx)]
        right = bin_edges[int(idx) + 1]
        return np.random.uniform(low=left, high=right)

    @staticmethod
    def _validate_feature(df: pd.DataFrame, feature: str):
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame columns.")
        numpy_dtype = df[feature].to_numpy().dtype
        if not np.issubdtype(numpy_dtype, np.number):
            raise TypeError(f"Feature '{feature}' must be numeric to be binned.")

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Discretizer must be fitted before calling transform or inverse_transform."
            )
