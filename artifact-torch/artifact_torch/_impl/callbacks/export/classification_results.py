from artifact_torch.base.components.callbacks.metadata import MetadataExportCallback


class ClassificationResultsExportCallback(MetadataExportCallback):
    _metadata_name = "CLASSIFICATION_RESULTS"

    @classmethod
    def _get_metadata_name(cls) -> str:
        return cls._metadata_name



