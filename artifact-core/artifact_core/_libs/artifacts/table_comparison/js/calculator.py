from typing import Dict, Mapping, Optional, Sequence, Set

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from artifact_core._utils.collections.sequence_concatenator import SequenceConcatenator


class JSDistanceCalculator:
    @classmethod
    def compute_mean_js(
        cls,
        df_real: pd.DataFrame,
        df_synthetic: pd.DataFrame,
        cts_features: Sequence[str],
        cat_features: Sequence[str],
        cat_unique_map: Mapping[str, Sequence[str]],
        n_bins_cts_histogram: int,
        categorical_only: bool,
    ) -> float:
        dict_js = cls.compute_dict_js(
            df_real=df_real,
            df_synthetic=df_synthetic,
            cts_features=cts_features,
            cat_features=cat_features,
            cat_unique_map=cat_unique_map,
            n_bins_cts_histogram=n_bins_cts_histogram,
            categorical_only=categorical_only,
        )
        mean_js_distance = np.mean(list(dict_js.values())).item()
        return mean_js_distance

    @classmethod
    def compute_dict_js(
        cls,
        df_real: pd.DataFrame,
        df_synthetic: pd.DataFrame,
        cts_features: Sequence[str],
        cat_features: Sequence[str],
        cat_unique_map: Mapping[str, Sequence[str]],
        n_bins_cts_histogram: int,
        categorical_only: bool,
    ) -> Mapping[str, float]:
        cls._validate_no_missing_cols(
            set_real_cols=set(df_real.columns),
            set_synthetic_cols=set(df_synthetic.columns),
            set_cat_features=set(cat_features),
            set_cts_features=set(cts_features),
        )
        if categorical_only:
            df_real_subset = df_real[cat_features].copy()
            df_synthetic_subset = df_synthetic[cat_features].copy()
            dict_js = cls._compute_dict_js_categorical(
                df_real=df_real_subset,
                df_synthetic=df_synthetic_subset,
                features=cat_features,
                cat_unique_map=cat_unique_map,
            )
        else:
            cls._validate_no_overlap(
                set_cat_features=set(cat_features),
                set_cts_features=set(cts_features),
            )
            bins_dict = cls._get_cts_feature_bins(
                df_real=df_real,
                df_synthetic=df_synthetic,
                cts_features=cts_features,
                n_bins=n_bins_cts_histogram,
            )
            df_real_discrete = cls._apply_bins_to_df(
                df=df_real,
                cts_features=cts_features,
                col_to_bins=bins_dict,
            )
            df_synthetic_discrete = cls._apply_bins_to_df(
                df=df_synthetic,
                cts_features=cts_features,
                col_to_bins=bins_dict,
            )
            dict_js = cls._compute_dict_js_categorical(
                df_real=df_real_discrete,
                df_synthetic=df_synthetic_discrete,
                features=SequenceConcatenator.concatenate(cat_features, cts_features),
                cat_unique_map=cat_unique_map,
            )
        return dict_js

    @classmethod
    def _compute_dict_js_categorical(
        cls,
        df_real: pd.DataFrame,
        df_synthetic: pd.DataFrame,
        features: Sequence[str],
        cat_unique_map: Mapping[str, Sequence[str]],
    ) -> Mapping[str, float]:
        dict_js = {}
        for feature in features:
            sr_real = df_real[feature].dropna().astype(str)
            sr_synth = df_synthetic[feature].dropna().astype(str)
            ls_unique_categories = cls._get_unique_categories(
                sr_real=sr_real,
                sr_synth=sr_synth,
                unique_categories=cat_unique_map.get(feature, None),
            )
            dist = cls._compute_js_categorical(
                sr_real=sr_real,
                sr_synth=sr_synth,
                unique_categories=ls_unique_categories,
            )
            dict_js[feature] = dist
        return dict_js

    @staticmethod
    def _get_cts_feature_bins(
        df_real: pd.DataFrame, df_synthetic: pd.DataFrame, cts_features: Sequence[str], n_bins: int
    ) -> Dict[str, np.ndarray]:
        df = pd.concat([df_real[cts_features], df_synthetic[cts_features]], axis=0)
        bins_dict = {}
        for feat in cts_features:
            _, bins = pd.cut(df[feat], bins=n_bins, include_lowest=True, retbins=True)
            bins_dict[feat] = bins
        return bins_dict

    @staticmethod
    def _apply_bins_to_df(
        df: pd.DataFrame,
        cts_features: Sequence[str],
        col_to_bins: Mapping[str, np.ndarray],
    ) -> pd.DataFrame:
        df = df.copy()
        for feat in cts_features:
            bins = pd.Series(col_to_bins[feat])
            df[feat] = pd.cut(x=df[feat], bins=bins, include_lowest=True)
        return df

    @staticmethod
    def _compute_js_categorical(
        sr_real: pd.Series,
        sr_synth: pd.Series,
        unique_categories: Sequence[str],
    ) -> float:
        sr_counts_real = sr_real.value_counts(normalize=True).reindex(
            list(unique_categories), fill_value=0
        )
        sr_counts_synthetic = sr_synth.value_counts(normalize=True).reindex(
            list(unique_categories), fill_value=0
        )
        dist = jensenshannon(p=sr_counts_real.values, q=sr_counts_synthetic.values)  # type: ignore
        return dist.item()

    @staticmethod
    def _get_unique_categories(
        sr_real: pd.Series,
        sr_synth: pd.Series,
        unique_categories: Optional[Sequence[str]] = None,
    ) -> Sequence[str]:
        if unique_categories is None:
            arr_real_values = sr_real.unique()
            arr_synthetic_values = sr_synth.unique()
            unique_categories = np.union1d(arr_real_values, arr_synthetic_values).tolist()  # type: ignore
            if unique_categories is None:
                unique_categories = []
        return unique_categories

    @staticmethod
    def _validate_no_missing_cols(
        set_real_cols: Set[str],
        set_synthetic_cols: Set[str],
        set_cts_features: Set[str],
        set_cat_features: Set[str],
    ):
        difference = set_cts_features.union(set_cat_features).difference(
            set_real_cols.intersection(set_synthetic_cols)
        )
        if difference:
            raise ValueError(f"Missing columns: {difference}.")

    @staticmethod
    def _validate_no_overlap(
        set_cts_features: Set[str],
        set_cat_features: Set[str],
    ):
        overlap = set_cts_features.intersection(set_cat_features)
        if set_cts_features.intersection(set_cat_features):
            raise ValueError(f"Continuous and categorical features overlap {overlap}")
