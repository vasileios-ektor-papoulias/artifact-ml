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
    _transformers_config = raw_config["transformers"]
    _architecture_config = raw_config["architecture"]
    _training_config = raw_config["training"]
    _validation_config = raw_config["validation"]

    # Data Config
    TRAINING_DATASET_PATH: Path = _artifact_torch_root / Path(_data_config["training_dataset_path"])
    LS_CTS_FEATURES: List[str] = _data_config["ls_cts_features"]
    LS_CAT_FEATURES: List[str] = _data_config["ls_cat_features"]

    # Transformers Config
    N_BINS_CTS: int = _transformers_config["n_bins_cts"]

    # Architecture Config
    N_EMBD: int = _architecture_config["n_embd"]
    LS_ENCODER_LAYER_SIZES: List[int] = _architecture_config["ls_encoder_layer_sizes"]
    LATENT_DIM: int = _architecture_config["latent_dim"]
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
    GENERATION_N_RECORDS: int = _validation_config["generation_n_records"]
    GENERATION_USE_MEAN: bool = _validation_config["generation_use_mean"]
    GENERATION_TEMPERATURE: float = _validation_config["generation_temperature"]
