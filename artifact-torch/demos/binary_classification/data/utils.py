import pandas as pd
from artifact_core.binary_classification import BinaryFeatureSpec
from artifact_core.libs.resources.categorical.category_store.binary import BinaryCategoryStore
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.binary_classification.routine import BinaryClassificationRoutineData

from demos.binary_classification.config.constants import BATCH_SIZE, DROP_LAST, LS_FEATURES, SHUFFLE
from demos.binary_classification.data.dataset import MLPClassifierDataset
from demos.binary_classification.model.protocols import MLPClassifierInput


class DemoDataUtils:
    @staticmethod
    def build_data_loader(
        df: pd.DataFrame, class_spec: BinaryFeatureSpec
    ) -> DataLoader[MLPClassifierInput]:
        return DataLoader(
            dataset=MLPClassifierDataset(
                df=df, ls_features=LS_FEATURES, label_feature=class_spec.feature_name
            ),
            batch_size=BATCH_SIZE,
            drop_last=DROP_LAST,
            shuffle=SHUFFLE,
        )

    @staticmethod
    def build_artifact_routine_data(
        df: pd.DataFrame, class_spec: BinaryFeatureSpec
    ) -> BinaryClassificationRoutineData[pd.DataFrame]:
        classification_data = df.drop(class_spec.feature_name, axis=1)
        id_to_category = df[class_spec.feature_name].to_dict()
        true_category_store = BinaryCategoryStore.from_categories_and_spec(
            feature_spec=class_spec, id_to_category=id_to_category
        )
        artifact_routine_data = BinaryClassificationRoutineData[pd.DataFrame](
            true_category_store=true_category_store, classification_data=classification_data
        )
        return artifact_routine_data
