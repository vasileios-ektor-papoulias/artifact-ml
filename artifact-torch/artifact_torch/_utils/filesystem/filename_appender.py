from pathlib import Path


class FilenameAppender:
    @staticmethod
    def append(filepath: str | Path, text: str) -> Path:
        p = Path(filepath)
        return p.with_name(f"{p.stem}{text}{p.suffix}")
