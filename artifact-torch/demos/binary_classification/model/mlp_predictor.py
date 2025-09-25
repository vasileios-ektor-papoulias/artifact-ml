from typing import Type, TypeVar

import torch
import torch.nn as nn
from demos.binary_classification.model.mlp_encoder import MLPEncoder, MLPEncoderConfig

MLPPredictorT = TypeVar("MLPPredictorT", bound="MLPPredictor")


class MLPPredictor(nn.Module):
    def __init__(self, encoder: MLPEncoder, head: nn.Linear, loss: nn.Module):
        super().__init__()
        self._encoder = encoder
        self._head = head
        self._loss = loss

    @classmethod
    def build(
        cls: Type[MLPPredictorT],
        n_classes: int,
        in_dim: int,
        config: MLPEncoderConfig = MLPEncoderConfig(),
    ) -> MLPPredictorT:
        encoder = MLPEncoder.build(in_dim=in_dim, config=config)
        head = nn.Linear(in_features=encoder.output_dim, out_features=n_classes)
        loss = nn.CrossEntropyLoss()
        model = cls(encoder=encoder, head=head, loss=loss)
        return model

    @property
    def input_dim(self) -> int:
        return self._encoder.input_dim

    @property
    def hidden_dim(self) -> int:
        return self._head.in_features

    @property
    def n_classes(self) -> int:
        return self._head.out_features

    def forward(self, t_features: torch.Tensor) -> torch.Tensor:
        t_hidden = self._encoder(t_features)
        logits = self._head(t_hidden)
        return logits

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._loss(logits, targets)

    def predict_proba(self, t_features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(t_features), dim=-1)

    def predict(self, t_features: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.predict_proba(t_features), dim=-1)
