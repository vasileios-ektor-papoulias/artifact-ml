from artifact_experiment._base.components.callbacks.metadata import MetadataExportCallback


class ClassificationExportCallback(MetadataExportCallback):
    _metadata_name = "CLASSIFICATION_RESULTS"

    @classmethod
    def _get_metadata_name(cls) -> str:
        return cls._metadata_name
