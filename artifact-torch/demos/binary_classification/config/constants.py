import json
from pathlib import Path
from typing import List

import torch

_artifact_torch_root = Path(__file__).resolve().parent.parent.parent.parent

_config_dir = _artifact_torch_root / "demos" / "binary_classification" / "config"
_config_file = _config_dir / "config.json"

with _config_file.open() as f:
    raw_config = json.load(f)

    # Config Categories
    _data_config = raw_config["data"]
    _architecture_config = raw_config["architecture"]
    _training_config = raw_config["training"]
    _validation_config = raw_config["validation"]
    _tracking_config = raw_config["tracking"]

    # Data Config
    TRAINING_DATASET_PATH: Path = _artifact_torch_root / Path(_data_config["training_dataset_path"])
    VAL_DATA_PROPORTION: float = _data_config["val_data_proportion"]
    LS_FEATURES: List[str] = _data_config["ls_features"]
    LABEL_FEATURE: str = _data_config["label_feature"]
    LS_CATEGORIES: List[str] = _data_config["ls_categories"]
    POSITIVE_CATEGORY: str = _data_config["positive_category"]

    # Architecture Config
    LS_HIDDEN_SIZES: List[int] = _architecture_config["ls_hidden_sizes"]
    LEAKY_RELU_SLOPE: float = _architecture_config["leaky_relu_slope"]
    BN_MOMENTUM: float = _architecture_config["bn_momentum"]
    BN_EPSILON: float = _architecture_config["bn_epsilon"]
    DROPOUT_RATE: float = _architecture_config["dropout_rate"]

    # Training Config
    DEVICE: torch.device = torch.device(_training_config["device"])
    MAX_N_EPOCHS: int = _training_config["max_n_epochs"]
    LEARNING_RATE: float = _training_config["learning_rate"]
    CHECKPOINT_PERIOD: int = _training_config["checkpoint_period"]
    BATCH_SIZE: int = _training_config["batch_size"]
    DROP_LAST: float = _training_config["drop_last"]
    SHUFFLE: bool = _training_config["shuffle"]

    # Validation Config
    TRAIN_DIAGNOSTICS_PERIOD: int = _validation_config["train_diagnostics_period"]
    LOADER_VALIDATION_PERIOD: int = _validation_config["loader_validation_period"]
    ARTIFACT_ROUTINE_PERIOD: int = _validation_config["artifact_routine_period"]
    CLASSIFICATION_THRESHOLD: float = _validation_config["classification_threshold"]

    # Tracking Config
    EXPERIMENT_ID: str = _tracking_config["experiment_id"]
