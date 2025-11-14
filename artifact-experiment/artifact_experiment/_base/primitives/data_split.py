from enum import Enum


class DataSplit(Enum):
    ALL = "ALL"
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"

    @property
    def suffix(self) -> str:
        if self is DataSplit.ALL:
            return ""
        elif self is DataSplit.TRAIN:
            return "TRAIN"
        elif self is DataSplit.VALIDATION:
            return "VAL"
        elif self is DataSplit.TEST:
            return "TEST"
        else:
            raise ValueError(f"Unrecognized DataSplit value: {self}")

    def append_to(self, name: str) -> str:
        suffix = self.suffix
        if not suffix:
            return name
        token = f"_{suffix}"
        return name if name.endswith(token) else f"{name}{token}"
