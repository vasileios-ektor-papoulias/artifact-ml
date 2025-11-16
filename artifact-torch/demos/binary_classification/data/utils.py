import pandas as pd
from artifact_torch.binary_classification import (
    BinaryClassificationRoutineData,
    BinaryClassSpec,
    BinaryClassStore,
)
from artifact_torch.nn import DataLoader

from demos.binary_classification.config.constants import BATCH_SIZE, DROP_LAST, LS_FEATURES, SHUFFLE
from demos.binary_classification.contracts.model import MLPClassifierInput
from demos.binary_classification.data.dataset import MLPClassifierDataset


class DemoDataUtils:
    @staticmethod
    def build_data_loader(
        df: pd.DataFrame, class_spec: BinaryClassSpec
    ) -> DataLoader[MLPClassifierInput]:
        return DataLoader(
            dataset=MLPClassifierDataset(
                df=df, ls_features=LS_FEATURES, label_feature=class_spec.label_name
            ),
            batch_size=BATCH_SIZE,
            drop_last=DROP_LAST,
            shuffle=SHUFFLE,
        )

    @staticmethod
    def build_artifact_routine_data(
        df: pd.DataFrame, class_spec: BinaryClassSpec
    ) -> BinaryClassificationRoutineData[pd.DataFrame]:
        classification_data = df.drop(class_spec.label_name, axis=1)
        id_to_class = df[class_spec.label_name].to_dict()
        true_class_store = BinaryClassStore.from_class_names_and_spec(
            class_spec=class_spec, id_to_class=id_to_class
        )
        artifact_routine_data = BinaryClassificationRoutineData[pd.DataFrame](
            true_class_store=true_class_store, classification_data=classification_data
        )
        return artifact_routine_data
