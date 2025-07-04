from artifact_core.libs.resource_spec.tabular.spec import TabularDataSpec
from artifact_core.libs.utils.package_importer import PackageImporter
from artifact_core.table_comparison.config.parsed import (
    ARTIFACT_CORE_ROOT,
    CUSTOM_ARTIFACT_PATH,
    NATIVE_ARTIFACT_PATH,
)
from artifact_core.table_comparison.engine.engine import (
    TableComparisonArrayCollectionType,
    TableComparisonArrayType,
    TableComparisonEngine,
    TableComparisonPlotCollectionType,
    TableComparisonPlotType,
    TableComparisonScoreCollectionType,
    TableComparisonScoreType,
)

if NATIVE_ARTIFACT_PATH is None:
    raise ValueError("Null native artifact path: edit the toolkit configuration file.")
PackageImporter.import_all_from_package_path(path=NATIVE_ARTIFACT_PATH, root=ARTIFACT_CORE_ROOT)
if CUSTOM_ARTIFACT_PATH is not None:
    PackageImporter.import_all_from_package_path(path=CUSTOM_ARTIFACT_PATH)
