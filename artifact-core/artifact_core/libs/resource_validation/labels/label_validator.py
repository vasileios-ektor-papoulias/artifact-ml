from typing import Dict, List

import pandas as pd


class LabelValidator:
    @classmethod
    def validate(
        cls,
        df: pd.DataFrame,
        id_col: str,
        label_classes_map: Dict[str, List[str]],
    ) -> pd.DataFrame:
        df = cls._validate_df(df)
        df = cls._validate_id_column(df, id_col)
        df = cls._validate_label_columns_exist(df, label_classes_map)
        df = cls._validate_label_categories(df, label_classes_map)
        cols = [id_col] + list(label_classes_map.keys())
        return df[cols].copy()

    @staticmethod
    def _validate_df(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("DataFrame must not be empty.")
        return df

    @staticmethod
    def _validate_id_column(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        if id_col not in df.columns:
            raise ValueError(f"ID column '{id_col}' not found in DataFrame.")
        s = df[id_col]
        if s.isna().any():
            raise ValueError(f"ID column '{id_col}' contains NaN values.")
        if s[s.notna()].duplicated(keep=False).any():
            raise ValueError(f"ID column '{id_col}' has duplicate values. IDs must be unique.")
        return df

    @staticmethod
    def _validate_label_columns_exist(
        df: pd.DataFrame,
        label_classes_map: Dict[str, List[str]],
    ) -> pd.DataFrame:
        missing = [label for label in label_classes_map.keys() if label not in df.columns]
        if missing:
            raise ValueError(f"Label columns not found in DataFrame: {missing}")
        return df

    @staticmethod
    def _validate_label_categories(
        df: pd.DataFrame,
        label_classes_map: Dict[str, List[str]],
    ) -> pd.DataFrame:
        for label, classes in label_classes_map.items():
            allowed = set(classes)
            mask = df[label].notna()
            if not mask.any():
                continue
            invalid = df.loc[mask, label][~df.loc[mask, label].isin(allowed)]
            if not invalid.empty:
                bad_examples = list(pd.unique(invalid))[:10]
                raise ValueError(
                    f"Column '{label}' has values outside its allowed classes. "
                    f"Examples: {bad_examples}. Allowed: {sorted(allowed)}"
                )
        return df
