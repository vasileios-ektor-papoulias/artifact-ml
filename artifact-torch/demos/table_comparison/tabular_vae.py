from dataclasses import dataclass
from typing import Optional, Type, TypeVar

import pandas as pd
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.libs.resource_spec.tabular.spec import TabularDataSpec
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_torch.base.data.data_loader import DataLoader
from demos.table_comparison.components.routines.artifact import DemoTableComparisonRoutine
from demos.table_comparison.config.constants import (
    BATCH_SIZE,
    DROP_LAST,
    GENERATION_TEMPERATURE,
    GENERATION_USE_MEAN,
    N_BINS_CTS,
    SHUFFLE,
)
from demos.table_comparison.data.dataset import TabularVAEDataset
from demos.table_comparison.libs.transformers.discretizer import Discretizer
from demos.table_comparison.libs.transformers.encoder import Encoder
from demos.table_comparison.model.synthesizer import (
    TabularVAEGenerationParams,
    TabularVAESynthesizer,
    VAEArchitectureConfig,
)
from demos.table_comparison.trainer.trainer import TabularVAETrainer


@dataclass
class TabularVAEConfig:
    n_bins_cts: int = N_BINS_CTS
    architecture: VAEArchitectureConfig = VAEArchitectureConfig()


TabularVAET = TypeVar("TabularVAET", bound="TabularVAE")


class TabularVAE:
    def __init__(self, discretizer: Discretizer, encoder: Encoder, config: TabularVAEConfig):
        self._discretizer = discretizer
        self._encoder = encoder
        self._config = config
        self._synthesizer: Optional[TabularVAESynthesizer] = None

    @classmethod
    def build(
        cls: Type[TabularVAET],
        data_spec: TabularDataSpecProtocol,
        config: TabularVAEConfig = TabularVAEConfig(),
    ) -> TabularVAET:
        discretizer = Discretizer(
            n_bins=config.n_bins_cts, ls_cts_features=data_spec.ls_cts_features
        )
        encoder = Encoder()
        interface = cls(discretizer=discretizer, encoder=encoder, config=config)
        return interface

    def fit(
        self,
        df: pd.DataFrame,
        data_spec: TabularDataSpecProtocol,
        tracking_client: Optional[TrackingClient] = None,
    ) -> pd.DataFrame:
        df_encoded = self._preprocess_training_data(df=df)
        data_spec_encoded = TabularDataSpec.from_df(
            df=df_encoded, ls_cat_features=list(df_encoded.columns)
        )
        self._synthesizer = TabularVAESynthesizer.build(
            data_spec=data_spec_encoded,
            discretizer=self._discretizer,
            encoder=self._encoder,
            architecture_config=self._config.architecture,
        )
        dataset = TabularVAEDataset(df_encoded=df_encoded)
        loader = DataLoader(
            dataset=dataset, batch_size=BATCH_SIZE, drop_last=DROP_LAST, shuffle=SHUFFLE
        )
        artifact_routine = DemoTableComparisonRoutine.build(
            df_real=df, data_spec=data_spec, tracking_client=tracking_client
        )
        trainer = TabularVAETrainer.build(
            model=self._synthesizer,
            train_loader=loader,
            artifact_routine=artifact_routine,
            tracking_client=tracking_client,
        )
        trainer.train()
        return trainer.epoch_scores

    def generate(self, n_records: int) -> pd.DataFrame:
        params = TabularVAEGenerationParams(
            n_records=n_records,
            use_mean=GENERATION_USE_MEAN,
            temperature=GENERATION_TEMPERATURE,
            sample=True,
        )
        assert self._synthesizer is not None, "Must fit before generating"
        return self._synthesizer.generate(params=params)

    def _preprocess_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self._discretizer.fit(df=df)
        df_discretized = self._discretizer.transform(df=df)
        self._encoder.fit(df=df_discretized, ls_cat_features=list(df_discretized.columns))
        df_encoded = self._encoder.transform(df=df_discretized)
        return df_encoded
