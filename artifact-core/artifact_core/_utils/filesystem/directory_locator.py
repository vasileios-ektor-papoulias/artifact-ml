from pathlib import Path
from typing import Optional


class DirectoryLocator:
    @staticmethod
    def find(marker: str, start: Optional[Path] = None) -> Optional[Path]:
        if start is None:
            start = Path.cwd()
        current = start.resolve()
        while current != current.parent:
            candidate = current / marker
            if candidate.exists() and candidate.is_dir():
                return candidate
            current = current.parent
        return None
