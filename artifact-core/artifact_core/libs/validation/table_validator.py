from typing import List

import pandas as pd


class TableValidator:
    @classmethod
    def validate(
        cls,
        df: pd.DataFrame,
        ls_features: List[str],
        ls_cts_features: List[str],
        ls_cat_features: List[str],
    ) -> pd.DataFrame:
        df = cls._validate_df(df=df, ls_features=ls_features)
        df = cls._validate_continuous_features(df=df, ls_cts_features=ls_cts_features)
        df = cls._validate_categorical_features(df=df, ls_cat_features=ls_cat_features)
        return df

    @staticmethod
    def _validate_df(
        df: pd.DataFrame,
        ls_features: List[str],
    ) -> pd.DataFrame:
        if df.empty:
            raise ValueError("DataFrame must not be empty.")
        return df[ls_features]

    @staticmethod
    def _validate_continuous_features(
        df: pd.DataFrame,
        ls_cts_features: List[str],
    ) -> pd.DataFrame:
        df = df.copy()
        for feature in ls_cts_features:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")
            if df[feature].dropna().empty:
                print(f"Warning: Continuous feature '{feature}' contains no numeric data.")
        return df

    @staticmethod
    def _validate_categorical_features(
        df: pd.DataFrame,
        ls_cat_features: List[str],
    ) -> pd.DataFrame:
        df = df.copy()
        for feature in ls_cat_features:
            df[feature] = df[feature].astype(str)
        return df
