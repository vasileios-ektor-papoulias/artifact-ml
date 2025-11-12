from pathlib import Path

from artifact_core._bootstrap.config.config import DomainToolkitConfig
from artifact_core._bootstrap.types.toolkit import DomainToolkit

TOOLKIT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILEPATH = TOOLKIT_ROOT / "_config" / "raw.json"

CONFIG = DomainToolkitConfig.from_json_file(
    filepath=CONFIG_FILEPATH,
    domain_toolkit=DomainToolkit.BINARY_CLASSIFICATION,
    domain_toolkit_root=TOOLKIT_ROOT,
)
