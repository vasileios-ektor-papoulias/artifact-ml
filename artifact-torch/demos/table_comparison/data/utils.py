import pandas as pd
from artifact_torch.core import DataLoader
from artifact_torch.table_comparison._routine import TableComparisonRoutineData

from demos.table_comparison.config.constants import BATCH_SIZE, DROP_LAST, SHUFFLE
from demos.table_comparison.contracts.model import TabularVAEInput
from demos.table_comparison.data.dataset import TabularVAEDataset
from demos.table_comparison.libs.transformers.discretizer import Discretizer
from demos.table_comparison.libs.transformers.encoder import Encoder


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

    @staticmethod
    def build_artifact_routine_data(df_real: pd.DataFrame) -> TableComparisonRoutineData:
        return TableComparisonRoutineData(df_real=df_real)
