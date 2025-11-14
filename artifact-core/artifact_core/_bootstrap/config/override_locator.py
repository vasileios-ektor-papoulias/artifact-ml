from pathlib import Path
from typing import Optional

from artifact_core._utils.filesystem.directory_locator import DirectoryLocator


class ConfigOverrideLocator:
    _override_dir_name = ".artifact-ml"

    @classmethod
    def find(cls) -> Optional[Path]:
        override_dir = DirectoryLocator.find(marker=cls._override_dir_name)
        if override_dir is not None:
            return override_dir
