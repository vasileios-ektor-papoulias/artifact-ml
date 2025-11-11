from artifact_core._libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core._libs.resource_spec.tabular.spec import TabularDataSpec
from artifact_core.table_comparison._engine.engine import (
    TableComparisonArrayCollectionType,
    TableComparisonArrayType,
    TableComparisonEngine,
    TableComparisonPlotCollectionType,
    TableComparisonPlotType,
    TableComparisonScoreCollectionType,
    TableComparisonScoreType,
)


def _init_toolkit():
    from artifact_core._bootstrap.initializer import ToolkitInitializer
    from artifact_core.table_comparison._config.parsed import CONFIG, TOOLKIT_ROOT

    ToolkitInitializer.initialize(
        domain_toolkit_root=TOOLKIT_ROOT,
        native_artifact_path=CONFIG.native_artifact_path,
        custom_artifact_path=CONFIG.custom_artifact_path,
    )


_init_toolkit()
del _init_toolkit
