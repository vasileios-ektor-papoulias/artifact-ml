from pathlib import Path
from typing import Optional

from artifact_core._utils.filesystem.directory_locator import DirectoryLocator


class ConfigOverrideLocator:
    _override_dir_name = ".artifact-ml"

    @classmethod
    def find(cls) -> Optional[Path]:
        return DirectoryLocator.find(marker=cls._override_dir_name)
