from artifact_core.binary_classification.config.parsed import (
    ARTIFACT_CORE_ROOT,
    CUSTOM_ARTIFACT_PATH,
    NATIVE_ARTIFACT_PATH,
)
from artifact_core.binary_classification.engine.engine import (
    BinaryClassificationArrayCollectionType,
    BinaryClassificationArrayType,
    BinaryClassificationEngine,
    BinaryClassificationPlotCollectionType,
    BinaryClassificationPlotType,
    BinaryClassificationScoreCollectionType,
    BinaryClassificationScoreType,
)
from artifact_core.libs.resource_spec.binary.spec import BinaryFeatureSpec
from artifact_core.libs.utils.system.package_importer import PackageImporter

if NATIVE_ARTIFACT_PATH is None:
    raise ValueError("Null native artifact path: edit the toolkit configuration file.")
PackageImporter.import_all_from_package_path(path=NATIVE_ARTIFACT_PATH, root=ARTIFACT_CORE_ROOT)
if CUSTOM_ARTIFACT_PATH is not None:
    PackageImporter.import_all_from_package_path(path=CUSTOM_ARTIFACT_PATH)
