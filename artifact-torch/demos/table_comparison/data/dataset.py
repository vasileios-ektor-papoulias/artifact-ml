from typing import TypeVar

import pandas as pd
import torch
from artifact_torch.base.data.dataset import Dataset
from demos.table_comparison.model.io import TabularVAEInput

TabularVAEDatasetT = TypeVar("TabularVAEDatasetT", bound="TabularVAEDataset")


class TabularVAEDataset(Dataset[TabularVAEInput]):
    def __init__(self, df_encoded: pd.DataFrame):
        self._t_data = torch.tensor(df_encoded.values, dtype=torch.float32)

    def __len__(self) -> int:
        return self._t_data.size(0)

    def __getitem__(self, idx: int) -> TabularVAEInput:
        row = self._t_data[idx]
        model_input = TabularVAEInput(t_features=row)
        return model_input
