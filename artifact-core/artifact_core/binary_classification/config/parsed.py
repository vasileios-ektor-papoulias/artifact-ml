import json
import os
from pathlib import Path
from typing import Any, Dict

from artifact_core.libs.utils.system.config_utils import (
    ConfigMerger,
    ConfigOverrideLocator,
    DomainToolkitConfigType,
)

ARTIFACT_CORE_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = Path(__file__).resolve().parent
CONFIG_FILE = CONFIG_DIR / "raw.json"

with CONFIG_FILE.open() as f:
    _dict_artifact_configs: Dict[str, Any] = json.load(f)

_user_override = ConfigOverrideLocator.get_config_override(
    domain_toolkit_config_type=DomainToolkitConfigType.TABLE_COMPARISON
)

_merged_artifact_configs = ConfigMerger.merge(
    base_config=_dict_artifact_configs, override=_user_override
)


NATIVE_ARTIFACT_PATH = os.path.join(
    ARTIFACT_CORE_ROOT, _merged_artifact_configs.get("native_artifact_path", "")
)
CUSTOM_ARTIFACT_PATH = _merged_artifact_configs.get("custom_artifact_path")
DICT_SCORES_CONFIG = _merged_artifact_configs.get("scores", {})
DICT_ARRAYS_CONFIG = _merged_artifact_configs.get("arrays", {})
DICT_PLOTS_CONFIG = _merged_artifact_configs.get("plots", {})
DICT_SCORE_COLLECTIONS_CONFIG = _merged_artifact_configs.get("score_collections", {})
DICT_ARRAY_COLLECTIONS_CONFIG = _merged_artifact_configs.get("array_collections", {})
DICT_PLOT_COLLECTIONS_CONFIG = _merged_artifact_configs.get("plot_collections", {})
