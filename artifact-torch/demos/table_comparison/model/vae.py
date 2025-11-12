from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, TypeVar

import torch
import torch.nn as nn
from artifact_core._libs.resources_spec.tabular.protocol import TabularDataSpecProtocol

from demos.table_comparison.config.constants import (
    BN_EPSILON,
    BN_MOMENTUM,
    DROPOUT_RATE,
    LATENT_DIM,
    LEAKY_RELU_SLOPE,
    LOSS_BETA,
    LS_ENCODER_LAYER_SIZES,
    N_EMBD,
)
from demos.table_comparison.libs.layers.diagonal_gaussian_latent import DiagonalGaussianLatent
from demos.table_comparison.libs.layers.embedder import MultiFeatureEmbedder
from demos.table_comparison.libs.layers.lin_bn_drop import LinBnDrop
from demos.table_comparison.libs.layers.mlp import MLP
from demos.table_comparison.libs.layers.multi_feature_predictor import MultiFeaturePredictor
from demos.table_comparison.libs.losses.beta_loss import BetaVAELoss
from demos.table_comparison.libs.utils.sampler import LogitSampler, SamplingStrategy


@dataclass(frozen=True)
class VAEArchitectureConfig:
    n_embd: int = N_EMBD
    ls_encoder_layer_sizes: List[int] = field(default_factory=lambda: LS_ENCODER_LAYER_SIZES.copy())
    latent_dim: int = LATENT_DIM
    loss_beta: float = LOSS_BETA
    leaky_relu_slope: float = LEAKY_RELU_SLOPE
    bn_momentum: float = BN_MOMENTUM
    bn_epsilon: float = BN_EPSILON
    dropout_rate: float = DROPOUT_RATE


VariationalAutoencoderT = TypeVar("VariationalAutoencoderT", bound="VariationalAutoencoder")


class VariationalAutoencoder(nn.Module):
    _sampling_strategy = SamplingStrategy.RANDOM_SAMPLE

    def __init__(
        self,
        embedder: MultiFeatureEmbedder,
        embedding_bridge: nn.Module,
        encoder: MLP,
        latent: DiagonalGaussianLatent,
        decoder: MLP,
        predictor: MultiFeaturePredictor,
        loss: BetaVAELoss,
    ):
        super().__init__()
        self._embedder = embedder
        self._embedding_bridge = embedding_bridge
        self._encoder = encoder
        self._latent = latent
        self._decoder = decoder
        self._predictor = predictor
        self._loss = loss

    @classmethod
    def build(
        cls: Type[VariationalAutoencoderT],
        data_spec: TabularDataSpecProtocol,
        config: VAEArchitectureConfig = VAEArchitectureConfig(),
    ) -> VariationalAutoencoderT:
        ls_n_cat_n_embd = cls._get_ls_n_cat_n_embd(
            dict_unique_counts=data_spec.cat_unique_count_map, n_embd=config.n_embd
        )
        model = cls._build(
            ls_n_cat_n_embd=ls_n_cat_n_embd,
            ls_encoder_layer_sizes=config.ls_encoder_layer_sizes,
            latent_dim=config.latent_dim,
            loss_beta=config.loss_beta,
            leaky_relu_slope=config.leaky_relu_slope,
            bn_momentum=config.bn_momentum,
            bn_epsilon=config.bn_epsilon,
            dropout_rate=config.dropout_rate,
        )
        return model

    @property
    def total_embedding_dim(self) -> int:
        return self._embedder.total_dim

    @property
    def input_dim(self) -> int:
        return self._encoder.ls_layer_sizes[0]

    @property
    def hidden_dim(self) -> int:
        return self._encoder.ls_layer_sizes[-1]

    @property
    def latent_dim(self) -> int:
        return self._latent.latent_dim

    def forward(
        self, t_features: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        t_embedded = self._embedder(t_features)
        t_input = self._embedding_bridge(t_embedded)
        t_hidden = self._encoder(t_input)
        t_latent_mean, t_latent_log_var = self._latent(t_hidden)
        t_latent = self._latent.reparameterize(t_latent_mean, t_latent_log_var)
        t_decoded = self._decoder(t_latent)
        ls_t_logits = self._predictor(t_decoded)
        t_loss = self._loss(
            ls_t_logits=ls_t_logits,
            t_targets=t_features,
            z_mean=t_latent_mean,
            z_log_var=t_latent_log_var,
        )
        return ls_t_logits, t_latent_mean, t_latent_log_var, t_loss

    def generate(
        self,
        n_records: int,
        use_mean: bool,
        temperature: float,
        device: torch.device,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            t_latent = self._latent.sample(
                num_samples=n_records,
                latent_dim=self.latent_dim,
                use_mean=use_mean,
                device=device,
            )
            t_decoded = self._decoder(t_latent)
            ls_t_logits = self._predictor(t_decoded)
            t_preds = LogitSampler.sample(
                ls_t_logits=ls_t_logits, temperature=temperature, strategy=self._sampling_strategy
            )
        return t_preds

    @staticmethod
    def _get_ls_n_cat_n_embd(
        dict_unique_counts: Dict[str, int],
        n_embd: int,
    ) -> List[Tuple[int, int]]:
        ls_n_cat_n_embd = [(n_cat, n_embd) for n_cat in dict_unique_counts.values()]
        return ls_n_cat_n_embd

    @classmethod
    def _build(
        cls: Type[VariationalAutoencoderT],
        ls_n_cat_n_embd: List[Tuple[int, int]],
        ls_encoder_layer_sizes: List[int],
        latent_dim: int,
        loss_beta: float,
        leaky_relu_slope: float,
        bn_momentum: float,
        bn_epsilon: float,
        dropout_rate: float,
    ) -> VariationalAutoencoderT:
        if len(ls_encoder_layer_sizes) < 1:
            raise ValueError("ls_encoder_layer_sizes must contain at least one hidden layer.")
        input_dim = ls_encoder_layer_sizes[0]
        embedder = MultiFeatureEmbedder(ls_n_cat_n_embd=ls_n_cat_n_embd)
        embedding_bridge = LinBnDrop(
            in_dim=embedder.total_dim,
            out_dim=input_dim,
            leaky_relu_slope=leaky_relu_slope,
            bn_momentum=bn_momentum,
            bn_epsilon=bn_epsilon,
            dropout_rate=dropout_rate,
        )
        encoder = MLP.build(
            ls_layer_sizes=ls_encoder_layer_sizes,
            leaky_relu_slope=leaky_relu_slope,
            bn_momentum=bn_momentum,
            bn_epsilon=bn_epsilon,
            dropout_rate=dropout_rate,
        )
        latent = DiagonalGaussianLatent(
            in_dim=ls_encoder_layer_sizes[-1],
            latent_dim=latent_dim,
        )
        decoder = MLP.build(
            ls_layer_sizes=[latent_dim] + list(reversed(ls_encoder_layer_sizes)),
            leaky_relu_slope=leaky_relu_slope,
            bn_momentum=bn_momentum,
            bn_epsilon=bn_epsilon,
            dropout_rate=dropout_rate,
        )
        predictor = MultiFeaturePredictor(
            input_dim=input_dim,
            ls_n_cat_n_embd=ls_n_cat_n_embd,
        )
        loss = BetaVAELoss(beta=loss_beta)
        return cls(
            embedder=embedder,
            embedding_bridge=embedding_bridge,
            encoder=encoder,
            latent=latent,
            decoder=decoder,
            predictor=predictor,
            loss=loss,
        )
