from typing import Type, TypeVar

import numpy as np
import pandas as pd
import torch
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_torch.base.data.dataset import Dataset

from demo.data.feature_flattener import FeatureFlattener
from demo.model.io import TabularVAEInput

TabularVAEDatasetT = TypeVar("TabularVAEDatasetT", bound="TabularVAEDataset")


class TabularVAEDataset(Dataset[TabularVAEInput]):
    def __init__(self, df: pd.DataFrame, flattener: FeatureFlattener):
        self._df = df.copy()
        self._flattener = flattener
        self._arr_flat = self._get_flattened_data()
        self._t_data = torch.tensor(self._arr_flat, dtype=torch.float32)

    @classmethod
    def build(
        cls: Type[TabularVAEDatasetT], df: pd.DataFrame, data_spec: TabularDataSpecProtocol
    ) -> TabularVAEDatasetT:
        flattener = FeatureFlattener(data_spec=data_spec)
        flattener.fit(df=df)
        dataset = cls(df=df, flattener=flattener)
        return dataset

    def __len__(self) -> int:
        return self._t_data.size(0)

    def __getitem__(self, idx: int) -> TabularVAEInput:
        row = self._t_data[idx]
        model_input = TabularVAEInput(t_features=row)
        return model_input

    def _get_flattened_data(self) -> np.ndarray:
        arr_flat = self._flattener.transform(df=self._df)
        return arr_flat
