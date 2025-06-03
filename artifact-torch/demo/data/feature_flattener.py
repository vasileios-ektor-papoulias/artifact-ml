from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class FeatureFlattener:
    def __init__(self, data_spec: TabularDataSpecProtocol):
        self._data_spec = data_spec
        self.original_cts_cols: List[str] = data_spec.ls_cts_features[:]
        self.original_cat_cols: List[str] = data_spec.ls_cat_features[:]

        self._cts_scaler = MinMaxScaler()
        self._cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        self._fitted = False
        self._flattened_cat_feature_names: Optional[List[str]] = None
        self._flattened_column_names: Optional[List[str]] = None

    @property
    def ls_original_column_names(self) -> List[str]:
        return self.original_cts_cols + self.original_cat_cols

    @property
    def ls_flattened_column_names(self) -> List[str]:
        if not self._fitted:
            raise RuntimeError("Must call .fit(df) before accessing ls_flattened_column_names.")
        assert self._flattened_column_names is not None
        return self._flattened_column_names

    def fit(self, df: pd.DataFrame) -> None:
        missing = set(self.ls_original_column_names) - set(df.columns)
        if missing:
            raise ValueError(f"[FeatureFlattener.fit] Missing columns in df: {missing!r}")

        tmp = df.copy()
        self._cast_cts_dtypes(df=tmp, dict_cts_dtypes=self._data_spec.dict_cts_dtypes)
        self._cast_cat_dtypes(
            df=tmp,
            dict_cat_dtypes=self._data_spec.dict_cat_dtypes,
            cat_unique_map=self._data_spec.cat_unique_map,
        )

        if self.original_cts_cols:
            self._cts_scaler.fit(tmp[self.original_cts_cols])

        if self.original_cat_cols:
            self._cat_encoder.fit(tmp[self.original_cat_cols])

        cts_names = self.original_cts_cols[:]
        if self.original_cat_cols:
            oh_names = self._cat_encoder.get_feature_names_out(self.original_cat_cols).tolist()
        else:
            oh_names = []

        self._flattened_cat_feature_names = oh_names
        self._flattened_column_names = cts_names + oh_names
        self._fitted = True

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureFlattener: call .fit(df) before .transform(...)")

        tmp = df.copy()
        self._cast_cts_dtypes(df=tmp, dict_cts_dtypes=self._data_spec.dict_cts_dtypes)
        self._cast_cat_dtypes(
            df=tmp,
            dict_cat_dtypes=self._data_spec.dict_cat_dtypes,
            cat_unique_map=self._data_spec.cat_unique_map,
        )

        parts: List[np.ndarray] = []

        if self.original_cts_cols:
            cts_arr = self._scale_cts(
                cts_scaler=self._cts_scaler,
                df=tmp,
                cts_cols=self.original_cts_cols,
            )
            parts.append(cts_arr)

        if self.original_cat_cols:
            cat_arr = self._encode_cat(
                cat_encoder=self._cat_encoder,
                df=tmp,
                cat_cols=self.original_cat_cols,
            )
            parts.append(cat_arr)
        arr_flat = self._hstack(parts=parts, n_rows=len(df))
        return arr_flat

    def inverse_transform(self, arr_flat: np.ndarray) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeatureFlattener: call .fit(df) before .inverse_transform(...)")

        n_cts = len(self.original_cts_cols)
        total_cat_dims = arr_flat.shape[1] - n_cts

        cts_part, cat_part = self._split_flat_array(
            flat_array=arr_flat,
            n_cts=n_cts,
            n_cat_oh=total_cat_dims,
        )

        if n_cts > 0:
            df_cts = self._inverse_cts(
                cts_scaler=self._cts_scaler,
                cts_part=cts_part,
                cts_cols=self.original_cts_cols,
            )
        else:
            df_cts = pd.DataFrame(data={}, columns=[])

        if total_cat_dims > 0:
            counts: List[int] = []
            for col in self.original_cat_cols:
                cnt = sum(
                    name.startswith(f"{col}_") for name in (self._flattened_cat_feature_names or [])
                )
                counts.append(cnt)
            df_cat = FeatureFlattener._argmax_to_labels(
                cat_array=cat_part,
                cat_cols=self.original_cat_cols,
                cat_unique_map=self._data_spec.cat_unique_map,
                counts=counts,
            )
        else:
            df_cat = pd.DataFrame(data={}, columns=[])

        return pd.concat([df_cts, df_cat], axis=1)[self.ls_original_column_names]

    @staticmethod
    def _cast_cts_dtypes(df: pd.DataFrame, dict_cts_dtypes: Dict[str, type]) -> None:
        for col, dtype in dict_cts_dtypes.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

    @staticmethod
    def _cast_cat_dtypes(
        df: pd.DataFrame, dict_cat_dtypes: Dict[str, type], cat_unique_map: Dict[str, List]
    ) -> None:
        for col, dtype in dict_cat_dtypes.items():
            if col not in df.columns:
                continue
            if col in cat_unique_map:
                cats = cat_unique_map[col]
                df[col] = pd.Categorical(df[col], categories=cats)
            else:
                df[col] = df[col].astype(dtype)

    @staticmethod
    def _scale_cts(cts_scaler: MinMaxScaler, df: pd.DataFrame, cts_cols: List[str]) -> np.ndarray:
        return cts_scaler.transform(df[cts_cols])

    @staticmethod
    def _encode_cat(
        cat_encoder: OneHotEncoder, df: pd.DataFrame, cat_cols: List[str]
    ) -> np.ndarray:
        raw = cat_encoder.transform(df[cat_cols])
        return np.asarray(raw)

    @staticmethod
    def _hstack(parts: List[np.ndarray], n_rows: int) -> np.ndarray:
        if not parts:
            return np.zeros((n_rows, 0), dtype=float)
        return np.hstack(parts)

    @staticmethod
    def _split_flat_array(
        flat_array: np.ndarray, n_cts: int, n_cat_oh: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if n_cts > 0:
            cts_part = flat_array[:, :n_cts]
        else:
            cts_part = np.zeros((flat_array.shape[0], 0))

        if n_cat_oh > 0:
            cat_part = flat_array[:, n_cts : n_cts + n_cat_oh]
        else:
            cat_part = np.zeros((flat_array.shape[0], 0))

        return cts_part, cat_part

    @staticmethod
    def _inverse_cts(
        cts_scaler: MinMaxScaler, cts_part: np.ndarray, cts_cols: List[str]
    ) -> pd.DataFrame:
        orig_cts = cts_scaler.inverse_transform(cts_part)
        return pd.DataFrame(orig_cts, columns=cts_cols)

    @staticmethod
    def _argmax_to_labels(
        cat_array: np.ndarray,
        cat_cols: List[str],
        cat_unique_map: Dict[str, List],
        counts: List[int],
    ) -> pd.DataFrame:
        splits: List[np.ndarray] = []
        start = 0
        for c in counts:
            end = start + c
            splits.append(cat_array[:, start:end])
            start = end

        decoded: Dict[str, List] = {}
        for col, block in zip(cat_cols, splits):
            idx = np.argmax(block, axis=1)
            categories = cat_unique_map[col]
            decoded[col] = [categories[i] for i in idx]

        df_cat = pd.DataFrame(decoded, columns=cat_cols)
        for col in cat_cols:
            if col in cat_unique_map:
                df_cat[col] = pd.Categorical(df_cat[col], categories=cat_unique_map[col])
        return df_cat
