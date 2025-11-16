import pandas as pd
import torch
from artifact_torch.nn import Dataset

from demos.table_comparison.contracts.model import TabularVAEInput
from demos.table_comparison.libs.transformers.discretizer import Discretizer
from demos.table_comparison.libs.transformers.encoder import Encoder


class TabularVAEDataset(Dataset[TabularVAEInput]):
    def __init__(self, df_raw: pd.DataFrame, discretizer: Discretizer, encoder: Encoder):
        df_discretized = discretizer.transform(df=df_raw)
        df_encoded = encoder.transform(df=df_discretized)
        self._t_data = torch.tensor(df_encoded.values, dtype=torch.float32)

    def __len__(self) -> int:
        return self._t_data.size(0)

    def __getitem__(self, idx: int) -> TabularVAEInput:
        row = self._t_data[idx]
        model_input = TabularVAEInput(t_features=row)
        return model_input
