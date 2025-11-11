from artifact_core._libs.resource_spec.binary.protocol import BinaryFeatureSpecProtocol
from artifact_core._libs.resource_spec.binary.spec import BinaryFeatureSpec
from artifact_core.binary_classification._engine.engine import (
    BinaryClassificationArrayCollectionType,
    BinaryClassificationArrayType,
    BinaryClassificationEngine,
    BinaryClassificationPlotCollectionType,
    BinaryClassificationPlotType,
    BinaryClassificationScoreCollectionType,
    BinaryClassificationScoreType,
)


def _init_toolkit():
    from artifact_core._bootstrap.initializer import ToolkitInitializer
    from artifact_core.binary_classification._config.parsed import CONFIG, TOOLKIT_ROOT

    ToolkitInitializer.initialize(
        domain_toolkit_root=TOOLKIT_ROOT,
        native_artifact_path=CONFIG.native_artifact_path,
        custom_artifact_path=CONFIG.custom_artifact_path,
    )


_init_toolkit()
del _init_toolkit
