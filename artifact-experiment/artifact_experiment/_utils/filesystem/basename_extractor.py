import os


class BasenameExtractor:
    @staticmethod
    def extract(path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]
