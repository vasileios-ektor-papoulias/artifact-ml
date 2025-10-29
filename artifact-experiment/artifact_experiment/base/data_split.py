from enum import Enum


class DataSplit(Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


class DataSplitSuffixAppender:
    @staticmethod
    def append_suffix(name: str, data_split: DataSplit) -> str:
        if data_split is DataSplit.TRAIN:
            return f"{name}_TRAIN"
        elif data_split is DataSplit.VALIDATION:
            return f"{name}_VAL"
        elif data_split is DataSplit.TEST:
            return f"{name}_TEST"
        else:
            raise ValueError(f"Unrecognized DataSplit value: {data_split}")
