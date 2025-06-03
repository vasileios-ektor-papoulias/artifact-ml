from dataclasses import dataclass
from typing import List, Type, TypeVar

import pandas as pd
from artifact_torch.core.model.generative import GenerationParams
from artifact_torch.table_comparison.model import TabularGenerativeModel

from demo.data.feature_flattener import FeatureFlattener
from demo.model.architecture import VariationalAutoencoder
from demo.model.io import TabularVAEInput, TabularVAEOutput


@dataclass
class TabularVAESynthesizerConfig:
    ls_encoder_layer_sizes: List[int]
    loss_beta: float
    leaky_relu_slope: float
    bn_momentum: float
    bn_epsilon: float
    dropout_rate: float


@dataclass
class TabularVAEGenerationParams(GenerationParams):
    num_samples: int
    use_mean: bool = False


TabularVAESynthesizerT = TypeVar("TabularVAESynthesizerT", bound="TabularVAESynthesizer")


class TabularVAESynthesizer(
    TabularGenerativeModel[TabularVAEInput, TabularVAEOutput, TabularVAEGenerationParams]
):
    def __init__(self, vae: VariationalAutoencoder, flattener: FeatureFlattener):
        super().__init__()
        self._vae = vae
        self._flattener = flattener

    @classmethod
    def build(
        cls: Type[TabularVAESynthesizerT],
        config: TabularVAESynthesizerConfig,
        flattener: FeatureFlattener,
    ) -> TabularVAESynthesizerT:
        vae = VariationalAutoencoder(
            ls_encoder_layer_sizes=config.ls_encoder_layer_sizes,
            loss_beta=config.loss_beta,
            leaky_relu_slope=config.leaky_relu_slope,
            bn_momentum=config.bn_momentum,
            bn_epsilon=config.bn_epsilon,
            dropout_rate=config.dropout_rate,
        )
        tabular_vae_model = cls(vae=vae, flattener=flattener)
        return tabular_vae_model

    @property
    def input_dim(self) -> int:
        return self._vae.input_dim

    @property
    def latent_dim(self) -> int:
        return self._vae.latent_dim

    @property
    def beta(self) -> float:
        return self._vae.loss_beta

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
        arr_synthetic_flat = self._vae.generate(
            num_samples=params.num_samples, use_mean=params.use_mean, device=self.device
        )
        arr_synthetic = self._flattener.inverse_transform(arr_flat=arr_synthetic_flat)
        df_synthetic = pd.DataFrame(arr_synthetic, columns=self._flattener.ls_original_column_names)
        return df_synthetic
