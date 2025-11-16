from typing import List, Type, TypeVar

import pandas as pd
import torch
from artifact_torch.core import Dataset

from demos.binary_classification.contracts.model import MLPClassifierInput

TabularVAEDatasetT = TypeVar("TabularVAEDatasetT", bound="MLPClassifierDataset")


class MLPClassifierDataset(Dataset[MLPClassifierInput]):
    def __init__(self, df: pd.DataFrame, ls_features: List[str], label_feature: str):
        self._t_features = torch.tensor(df[ls_features].to_numpy(), dtype=torch.float32)
        self._t_targets = torch.tensor(df[label_feature].to_numpy(), dtype=torch.long)

    @classmethod
    def build(
        cls: Type[TabularVAEDatasetT],
        df: pd.DataFrame,
        ls_features: List[str],
        label_feature: str,
    ) -> TabularVAEDatasetT:
        cls._validate_data(df=df, ls_features=ls_features, label_feature=label_feature)
        dataset = cls(df=df, ls_features=ls_features, label_feature=label_feature)
        return dataset

    def __len__(self) -> int:
        return self._t_features.size(0)

    def __getitem__(self, idx: int) -> MLPClassifierInput:
        t_features = self._t_features[idx]
        t_targets = self._t_targets[idx]
        model_input = MLPClassifierInput(t_features=t_features, t_targets=t_targets)
        return model_input

    @staticmethod
    def _validate_data(df: pd.DataFrame, ls_features: List[str], label_feature: str):
        missing = [c for c in ls_features + [label_feature] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataframe: {missing}")
