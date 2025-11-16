from typing import Type, TypeVar

import pandas as pd
from artifact_torch.table_comparison import TabularDataSpec
from artifact_torch.table_comparison._model import TableSynthesizer

from demos.table_comparison.contracts.model import (
    TabularVAEGenerationParams,
    TabularVAEInput,
    TabularVAEOutput,
)
from demos.table_comparison.libs.transformers.discretizer import Discretizer
from demos.table_comparison.libs.transformers.encoder import Encoder
from demos.table_comparison.model.vae import VAEArchitectureConfig, VariationalAutoencoder

TabularVAESynthesizerT = TypeVar("TabularVAESynthesizerT", bound="TabularVAESynthesizer")


class TabularVAESynthesizer(
    TableSynthesizer[TabularVAEInput, TabularVAEOutput, TabularVAEGenerationParams]
):
    _use_mean = False
    _sample = True

    def __init__(
        self,
        data_spec: TabularDataSpec,
        discretizer: Discretizer,
        encoder: Encoder,
        vae: VariationalAutoencoder,
    ):
        super().__init__()
        self._data_spec = data_spec
        self._discretizer = discretizer
        self._encoder = encoder
        self._vae = vae

    @classmethod
    def build(
        cls: Type[TabularVAESynthesizerT],
        data_spec: TabularDataSpec,
        discretizer: Discretizer,
        encoder: Encoder,
        architecture_config: VAEArchitectureConfig = VAEArchitectureConfig(),
    ) -> TabularVAESynthesizerT:
        vae = VariationalAutoencoder.build(data_spec=data_spec, config=architecture_config)
        synthesizer = cls(
            data_spec=data_spec,
            discretizer=discretizer,
            encoder=encoder,
            vae=vae,
        )
        return synthesizer

    def forward(self, model_input: TabularVAEInput) -> TabularVAEOutput:
        t_features = model_input.get("t_features")
        ls_t_logits, t_latent_mean, t_latent_log_var, t_loss = self._vae(t_features=t_features)
        model_output = TabularVAEOutput(
            ls_t_logits=ls_t_logits,
            t_latent_mean=t_latent_mean,
            t_latent_log_var=t_latent_log_var,
            t_loss=t_loss,
        )
        return model_output

    def generate(self, params: TabularVAEGenerationParams) -> pd.DataFrame:
        self.eval()
        t_preds = self._vae.generate(
            n_records=params["n_records"],
            use_mean=self._use_mean,
            temperature=params["temperature"],
            device=self.device,
        )
        df_synthetic_encoded = pd.DataFrame(
            t_preds.cpu().numpy(), columns=list(self._data_spec.features)
        ).astype(int)
        df_synthetic = self._encoder.inverse_transform(df_encoded=df_synthetic_encoded)
        if self._sample:
            df_synthetic = self._discretizer.inverse_transform(df_binned=df_synthetic)
        return df_synthetic
