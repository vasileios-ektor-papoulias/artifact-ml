class ExtensionNormalizer:
    @staticmethod
    def normalize(extension: str) -> str:
        if not extension.startswith("."):
            return f".{extension}"
        return extension
