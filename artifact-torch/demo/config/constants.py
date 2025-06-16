import json
from pathlib import Path
from typing import List

import torch

_artifact_torch_root = Path(__file__).resolve().parent.parent.parent

_config_dir = _artifact_torch_root / "demo" / "config"
_config_file = _config_dir / "config.json"

with _config_file.open() as f:
    raw_config = json.load(f)

    # Config Categories
    _data_config = raw_config["data"]
    _architecture_config = raw_config["architecture"]
    _training_config = raw_config["training"]
    _validation_config = raw_config["validation"]

    # Data Config
    TRAINING_DATASET_PATH: Path = _artifact_torch_root / Path(_data_config["training_dataset_path"])

    # Architecture Config
    LS_ENCODER_LAYER_SIZES: List[int] = _architecture_config["ls_encoder_layer_sizes"]
    LOSS_BETA: float = _architecture_config["loss_beta"]
    LEAKY_RELU_SLOPE: float = _architecture_config["leaky_relu_slope"]
    BN_MOMENTUM: float = _architecture_config["bn_momentum"]
    BN_EPSILON: float = _architecture_config["bn_epsilon"]
    DROPOUT_RATE: float = _architecture_config["dropout_rate"]

    # Training Config
    DEVICE: torch.device = torch.device(_training_config["device"])
    MAX_N_EPOCHS: int = _training_config["max_n_epochs"]
    LEARNING_RATE: float = _training_config["learning_rate"]
    CHECKPOINT_PERIOD: int = _training_config["checkpoint_period"]
    BATCH_LOSS_PERIOD: int = _training_config["batch_loss_period"]
    BATCH_SIZE: int = _training_config["batch_size"]
    DROP_LAST: float = _training_config["drop_last"]
    SHUFFLE: bool = _training_config["shuffle"]

    # Validation Config
    TRAIN_LOADER_CALLBACK_PERIOD: int = _validation_config["train_loader_callback_period"]
    ARTIFACT_VALIDATION_PERIOD: int = _validation_config["validation_plan_callback_period"]
    GENERATION_NUM_SAMPLES: int = _validation_config["generation_num_samples"]
    GENERATION_USE_MEAN: bool = _validation_config["generation_use_mean"]
