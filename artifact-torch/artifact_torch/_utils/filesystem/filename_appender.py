from pathlib import Path


class FilenameModifier:
    @staticmethod
    def append(filepath: str | Path, text: str) -> Path:
        p = Path(filepath)
        return p.with_name(f"{p.stem}{text}{p.suffix}")

    @staticmethod
    def replace(filepath: str | Path, new_basename: str) -> Path:
        p = Path(filepath)
        return p.with_name(f"{new_basename}{p.suffix}")
