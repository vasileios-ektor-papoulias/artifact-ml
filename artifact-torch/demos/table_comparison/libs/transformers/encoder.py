from typing import Dict, List, Mapping, Optional, TypeVar

import numpy as np
import pandas as pd

EncoderT = TypeVar("EncoderT", bound="Encoder")


class Encoder:
    def __init__(self, cat_unique_map: Optional[Mapping[str, List[str]]] = None):
        self._cat_unique_map: Dict[str, List[str]] = (
            dict(cat_unique_map) if cat_unique_map is not None else {}
        )
        self._ls_cat_features: List[str] = list(self._cat_unique_map.keys())
        self._mappings: Dict[str, Dict[str, int]] = {}
        self._is_fitted: bool = False

    def get_mappings(self) -> Mapping[str, Mapping[str, int]]:
        self._raise_if_not_fitted()
        return self._mappings

    def fit(
        self: EncoderT, df: pd.DataFrame, ls_cat_features: Optional[List[str]] = None
    ) -> EncoderT:
        self._raise_if_fitted()
        if ls_cat_features is not None:
            self._ls_cat_features = ls_cat_features
            self._cat_unique_map = {}

        for feature in self._ls_cat_features:
            self._validate_feature(df=df, feature=feature)
            self._cat_unique_map[feature] = self._get_or_extract_categories(df=df, feature=feature)
            self._mappings[feature] = self._build_mapping_from_categories(
                ls_categories=self._cat_unique_map[feature]
            )
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._raise_if_not_fitted()
        return self._transform(df)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()
        for feature in self._ls_cat_features:
            self._validate_feature(df=df_encoded, feature=feature)
            if feature not in self._mappings:
                raise RuntimeError(
                    f"No mapping found for feature '{feature}'. Ensure `fit()` was called."
                )
            mapping = self._mappings[feature]
            df_encoded[feature] = df_encoded[feature].astype(str)
            self._validate_no_unknowns(
                values=df_encoded[feature],
                known=set(mapping.keys()),
                context="categories",
                feature=feature,
            )
            df_encoded[feature] = df_encoded[feature].map(mapping)
        return df_encoded

    def inverse_transform(self, df_encoded: pd.DataFrame) -> pd.DataFrame:
        self._raise_if_not_fitted()
        return self._inverse_transform(df_encoded)

    def _inverse_transform(self, df_encoded: pd.DataFrame) -> pd.DataFrame:
        df_decoded = df_encoded.copy()
        for feature, mapping in self._mappings.items():
            if feature not in df_decoded.columns:
                raise ValueError(
                    f"Column '{feature}' not found in DataFrame for inverse transformation."
                )
            inverse_mapping = self._invert_mapping(mapping)
            df_decoded[feature] = df_decoded[feature].astype(int)
            self._validate_no_unknowns(
                values=df_decoded[feature],
                known=set(inverse_mapping.keys()),
                context="encoded values",
                feature=feature,
            )
            df_decoded[feature] = df_decoded[feature].map(inverse_mapping)
        return df_decoded

    def _get_or_extract_categories(self, df: pd.DataFrame, feature: str) -> List[str]:
        if feature in self._cat_unique_map:
            return self._cat_unique_map[feature]
        return self._extract_unique_values(df[feature])

    @staticmethod
    def _build_mapping_from_categories(ls_categories: List[str]) -> Dict[str, int]:
        return {val: idx for idx, val in enumerate(ls_categories)}

    @staticmethod
    def _extract_unique_values(series: pd.Series) -> List[str]:
        return [str(x) for x in sorted(series.dropna().astype(str).unique())]

    @staticmethod
    def _invert_mapping(mapping: Dict[str, int]) -> Dict[int, str]:
        return {v: k for k, v in mapping.items()}

    @staticmethod
    def _validate_feature(df: pd.DataFrame, feature: str) -> None:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame columns.")

    @staticmethod
    def _validate_no_unknowns(values: pd.Series, known: set, context: str, feature: str) -> None:
        unknowns = set(values.dropna().unique())
        unknowns = {
            int(x) if isinstance(x, (np.integer, float)) and x == int(x) else x for x in unknowns
        }
        unknowns -= known
        if unknowns:
            raise ValueError(f"Unseen {context} in feature '{feature}': {unknowns}")

    def _raise_if_not_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Encoder must be fitted before calling transform or inverse_transform."
            )

    def _raise_if_fitted(self) -> None:
        if self._is_fitted:
            raise RuntimeError("Encoder is already fitted.")
