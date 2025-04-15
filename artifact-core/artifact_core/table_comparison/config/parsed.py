import json
from pathlib import Path
from typing import Any, Dict

from artifact_core.libs.utils.config_utils import (
    ConfigMerger,
    ConfigOverrideLocator,
    EngineConfigType,
)

CONFIG_DIR = Path(__file__).resolve().parent
CONFIG_FILE = CONFIG_DIR / "raw.json"

with CONFIG_FILE.open() as f:
    _dict_artifact_configs: Dict[str, Any] = json.load(f)

_user_override = ConfigOverrideLocator.get_config_override(
    engine_config_type=EngineConfigType.TABLE_COMPARISON
)

_merged_artifact_configs = ConfigMerger.merge(
    base_config=_dict_artifact_configs, override=_user_override
)

DICT_SCORES_CONFIG = _merged_artifact_configs.get("scores", {})
DICT_ARRAYS_CONFIG = _merged_artifact_configs.get("arrays", {})
DICT_PLOTS_CONFIG = _merged_artifact_configs.get("plots", {})
DICT_SCORE_COLLECTIONS_CONFIG = _merged_artifact_configs.get("score_collections", {})
DICT_ARRAY_COLLECTIONS_CONFIG = _merged_artifact_configs.get("array_collections", {})
DICT_PLOT_COLLECTIONS_CONFIG = _merged_artifact_configs.get("plot_collections", {})
