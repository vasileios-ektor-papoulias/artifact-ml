from typing import List, Optional, Type, TypeVar

import pandas as pd
from artifact_core._libs.resource_spec.binary.protocol import BinaryFeatureSpecProtocol
from artifact_core._libs.resources.categorical.category_store.binary import BinaryCategoryStore
from artifact_core._libs.resources.classification.binary_classification_results import (
    BinaryClassificationResults,
)
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_torch.base.data.data_loader import DataLoader

from demos.binary_classification.components.routines.artifact import DemoBinaryClassificationRoutine
from demos.binary_classification.config.constants import (
    BATCH_SIZE,
    CLASSIFICATION_THRESHOLD,
    DROP_LAST,
    SHUFFLE,
)
from demos.binary_classification.data.dataset import MLPClassifierDataset
from demos.binary_classification.model.classifier import MLPClassificationParams, MLPClassifierModel
from demos.binary_classification.model.mlp_encoder import MLPEncoderConfig
from demos.binary_classification.trainer.trainer import MLPClassifierTrainer

MLPClassifierT = TypeVar("MLPClassifierT", bound="MLPClassifier")


class MLPClassifier:
    def __init__(self, classifier: MLPClassifierModel):
        self._classifier = classifier

    @classmethod
    def build(
        cls: Type[MLPClassifierT],
        class_spec: BinaryFeatureSpecProtocol,
        ls_features: List[str],
        architecture_config: MLPEncoderConfig = MLPEncoderConfig(),
    ) -> MLPClassifierT:
        classifier = MLPClassifierModel.build(
            class_spec=class_spec, ls_features=ls_features, architecture_config=architecture_config
        )
        interface = cls(classifier=classifier)
        return interface

    def fit(
        self,
        df: pd.DataFrame,
        class_spec: BinaryFeatureSpecProtocol,
        ls_features: List[str],
        tracking_client: Optional[TrackingClient] = None,
    ) -> pd.DataFrame:
        self._classifier = MLPClassifierModel.build(class_spec=class_spec, ls_features=ls_features)
        dataset = MLPClassifierDataset(
            df=df, ls_features=ls_features, label_feature=class_spec.feature_name
        )
        loader = DataLoader(
            dataset=dataset, batch_size=BATCH_SIZE, drop_last=DROP_LAST, shuffle=SHUFFLE
        )
        id_to_category = df[class_spec.feature_name].to_dict()
        true_category_store = BinaryCategoryStore.from_categories_and_spec(
            feature_spec=class_spec, id_to_category=id_to_category
        )
        classification_data = df.drop(class_spec.feature_name, axis=1)
        artifact_routine = DemoBinaryClassificationRoutine.build(
            true_category_store=true_category_store,
            classification_data=classification_data,
            class_spec=class_spec,
            tracking_client=tracking_client,
        )
        trainer = MLPClassifierTrainer.build(
            model=self._classifier,
            train_loader=loader,
            artifact_routine=artifact_routine,
            tracking_client=tracking_client,
        )
        trainer.train()
        return trainer.epoch_scores

    def classify(self, df: pd.DataFrame) -> BinaryClassificationResults:
        params = MLPClassificationParams(threshold=CLASSIFICATION_THRESHOLD)
        classification_results = self._classifier.classify(data=df, params=params)
        return classification_results
