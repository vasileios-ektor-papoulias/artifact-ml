from pathlib import Path
from typing import Optional, Union, overload


class PathResolver:
    @overload
    @staticmethod
    def resolve(path: Union[Path, str]) -> str: ...

    @overload
    @staticmethod
    def resolve(path: None) -> None: ...

    @staticmethod
    def resolve(path: Optional[Union[Path, str]]) -> Optional[str]:
        if path is not None:
            return str(Path(path).resolve())
