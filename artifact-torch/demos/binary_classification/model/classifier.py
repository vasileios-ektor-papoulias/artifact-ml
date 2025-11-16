from typing import Dict, Hashable, List, Tuple, Type, TypeVar

import pandas as pd
import torch
from artifact_torch.binary_classification import (
    BinaryClassificationResults,
    BinaryClassifier,
    BinaryClassSpec,
)

from demos.binary_classification.contracts.model import (
    MLPClassificationParams,
    MLPClassifierInput,
    MLPClassifierOutput,
)
from demos.binary_classification.model.mlp_encoder import MLPEncoderConfig
from demos.binary_classification.model.mlp_predictor import MLPPredictor

MLPClassifierT = TypeVar("MLPClassifierT", bound="MLPClassifier")


class MLPClassifier(
    BinaryClassifier[MLPClassifierInput, MLPClassifierOutput, MLPClassificationParams, pd.DataFrame]
):
    _n_classes = 2

    def __init__(
        self,
        class_spec: BinaryClassSpec,
        ls_features: List[str],
        mlp: MLPPredictor,
    ):
        super().__init__()
        self._class_spec = class_spec
        self._ls_features = ls_features
        self._mlp = mlp

    @classmethod
    def build(
        cls: Type[MLPClassifierT],
        class_spec: BinaryClassSpec,
        ls_features: List[str],
        architecture_config: MLPEncoderConfig = MLPEncoderConfig(),
    ) -> MLPClassifierT:
        if not ls_features:
            raise ValueError(f"ls_features must be a nonempty list, got: {ls_features}.")
        mlp = MLPPredictor.build(
            in_dim=len(ls_features), n_classes=cls._n_classes, config=architecture_config
        )
        classifier = cls(
            class_spec=class_spec,
            ls_features=ls_features,
            mlp=mlp,
        )
        return classifier

    def forward(self, model_input: MLPClassifierInput) -> MLPClassifierOutput:
        t_features = model_input["t_features"]
        t_targets = model_input["t_targets"]
        t_logits = self._mlp(t_features)
        t_loss = self._mlp.compute_loss(t_logits, t_targets.long())
        model_output = MLPClassifierOutput(t_logits=t_logits, t_loss=t_loss)
        return model_output

    def classify(
        self, data: pd.DataFrame, params: MLPClassificationParams
    ) -> BinaryClassificationResults:
        t_features = self._to_t_features(df=data)  # (B, in_dim)
        t_prob_pos = self._infer_prob_pos(t_features=t_features)  # (B,)
        t_preds_bin = self._apply_threshold(
            t_prob_pos=t_prob_pos, threshold=params["threshold"]
        )  # (B,)
        id_to_class, id_to_prob_pos = self._build_classification_result_maps(
            data_index=data.index.tolist(),
            t_preds_bin=t_preds_bin,
            t_prob_pos=t_prob_pos,
        )
        classification_results = self._build_classification_results(
            id_to_class=id_to_class, id_to_prob_pos=id_to_prob_pos
        )
        return classification_results

    def _to_t_features(self, df: pd.DataFrame) -> torch.Tensor:
        missing = [c for c in self._ls_features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        x_np = df[self._ls_features].to_numpy(dtype="float32", copy=False)
        t_features = torch.from_numpy(x_np)
        return t_features

    @torch.no_grad()
    def _infer_prob_pos(self, t_features: torch.Tensor) -> torch.Tensor:
        t_logits = self._mlp(t_features)  # (B, C)
        t_probs_full = torch.softmax(t_logits, dim=-1)  # (B, C)
        return t_probs_full[..., self._class_spec.positive_class_idx]

    @staticmethod
    def _apply_threshold(t_prob_pos: torch.Tensor, threshold: float) -> torch.Tensor:
        thr = float(threshold)
        if not (0.0 <= thr <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        return t_prob_pos >= thr

    def _build_classification_result_maps(
        self,
        data_index: List[Hashable],
        t_preds_bin: torch.Tensor,
        t_prob_pos: torch.Tensor,
    ) -> Tuple[Dict[Hashable, str], Dict[Hashable, float]]:
        pos = self._class_spec.positive_class
        neg = self._class_spec.negative_class
        pred_labels = [pos if b.item() else neg for b in t_preds_bin]
        id_to_class = {idx: label for idx, label in zip(data_index, pred_labels)}
        id_to_prob_pos = {idx: float(p) for idx, p in zip(data_index, t_prob_pos.tolist())}
        return id_to_class, id_to_prob_pos

    def _build_classification_results(
        self,
        id_to_class: Dict,
        id_to_prob_pos: Dict,
    ) -> BinaryClassificationResults:
        pos = self._class_spec.positive_class
        class_names = self._class_spec.class_names
        classification_results = BinaryClassificationResults.build(
            class_names=class_names,
            positive_class=pos,
            id_to_class=id_to_class,
            id_to_prob_pos=id_to_prob_pos,
        )
        return classification_results
