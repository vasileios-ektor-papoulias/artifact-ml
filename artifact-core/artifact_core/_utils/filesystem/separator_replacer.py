import os
from pathlib import Path
from typing import Union


class SeparatorReplacer:
    @staticmethod
    def replace_separator(path: Union[Path, str], new: str) -> str:
        path_str = str(path) if isinstance(path, Path) else path
        return path_str.replace(os.sep, new)
