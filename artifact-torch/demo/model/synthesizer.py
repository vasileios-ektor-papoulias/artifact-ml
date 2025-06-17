from dataclasses import dataclass
from typing import Type, TypeVar

import pandas as pd
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_torch.core.model.generative import GenerationParams
from artifact_torch.table_comparison.model import TableSynthesizer

from demo.libs.transformers.discretizer import Discretizer
from demo.libs.transformers.encoder import Encoder
from demo.model.io import TabularVAEInput, TabularVAEOutput
from demo.model.vae import VAEArchitectureConfig, VariationalAutoencoder


@dataclass
class TabularVAEGenerationParams(GenerationParams):
    n_records: int
    use_mean: bool
    temperature: float
    sample: bool


TabularVAESynthesizerT = TypeVar("TabularVAESynthesizerT", bound="TabularVAESynthesizer")


class TabularVAESynthesizer(
    TableSynthesizer[TabularVAEInput, TabularVAEOutput, TabularVAEGenerationParams]
):
    def __init__(
        self,
        data_spec: TabularDataSpecProtocol,
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
        data_spec: TabularDataSpecProtocol,
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
        t_recon, z_mean, z_log_var, t_loss = self._vae(t_features=t_features)
        model_output = TabularVAEOutput(
            t_reconstructions=t_recon,
            z_mean=z_mean,
            z_log_var=z_log_var,
            t_loss=t_loss,
        )
        return model_output

    def generate(self, params: TabularVAEGenerationParams) -> pd.DataFrame:
        self.eval()
        t_preds = self._vae.generate(
            n_records=params.n_records,
            use_mean=params.use_mean,
            temperature=params.temperature,
            device=self.device,
        )
        df_synthetic_encoded = pd.DataFrame(
            t_preds.cpu().numpy(), columns=self._data_spec.ls_features
        ).astype(int)
        df_synthetic = self._encoder.inverse_transform(df_encoded=df_synthetic_encoded)
        if params.sample:
            df_synthetic = self._discretizer.inverse_transform(df_binned=df_synthetic)

        return df_synthetic
