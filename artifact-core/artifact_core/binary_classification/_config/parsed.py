from pathlib import Path

from artifact_core._bootstrap.config import ToolkitConfig
from artifact_core._bootstrap.toolkit import DomainToolkit

TOOLKIT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILEPATH = TOOLKIT_ROOT / "_config" / "raw.json"

CONFIG = ToolkitConfig.from_json_file(
    filepath=CONFIG_FILEPATH,
    domain_toolkit=DomainToolkit.BINARY_CLASSIFICATION,
    domain_toolkit_root=TOOLKIT_ROOT,
)
