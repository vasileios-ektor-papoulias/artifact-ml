from pathlib import Path

from artifact_core.libs.utils.package_importer import PackageImporter

from tests.base.dummy.artifact_dependencies import (
    DummyArtifactResources,
    DummyResourceSpec,
)
from tests.base.dummy.registries import (
    DummyArrayCollectionRegistry,
    DummyArrayCollectionType,
    DummyArrayRegistry,
    DummyArrayType,
    DummyPlotCollectionRegistry,
    DummyPlotCollectionType,
    DummyPlotRegistry,
    DummyPlotType,
    DummyScoreCollectionRegistry,
    DummyScoreCollectionType,
    DummyScoreRegistry,
    DummyScoreType,
)

PackageImporter.import_all_from_package_path(path=Path(__file__).resolve().parent)
print("banlet")
