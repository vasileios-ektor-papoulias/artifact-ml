from artifact_core._bootstrap.types.toolkit import DomainToolkit


class ConfigFilenameResolver:
    @staticmethod
    def get_config_filename(domain_toolkit: DomainToolkit) -> str:
        if domain_toolkit is DomainToolkit.TABLE_COMPARISON:
            return "table_comparison.json"
        if domain_toolkit is DomainToolkit.BINARY_CLASSIFICATION:
            return "binary_classification.json"
        else:
            raise ValueError(f"Unrecognized domain toolkit: {domain_toolkit=}")
