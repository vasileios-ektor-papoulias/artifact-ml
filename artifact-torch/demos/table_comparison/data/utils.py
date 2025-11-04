import pandas as pd
from artifact_torch.base.data.data_loader import DataLoader

from demos.table_comparison.config.constants import BATCH_SIZE, DROP_LAST, SHUFFLE
from demos.table_comparison.data.dataset import TabularVAEDataset
from demos.table_comparison.libs.transformers.discretizer import Discretizer
from demos.table_comparison.libs.transformers.encoder import Encoder
from demos.table_comparison.model.protocols import TabularVAEInput


class DemoDataUtils:
    @staticmethod
    def build_data_loader(
        df: pd.DataFrame, discretizer: Discretizer, encoder: Encoder
    ) -> DataLoader[TabularVAEInput]:
        return DataLoader(
            dataset=TabularVAEDataset(df_raw=df, discretizer=discretizer, encoder=encoder),
            batch_size=BATCH_SIZE,
            drop_last=DROP_LAST,
            shuffle=SHUFFLE,
        )
