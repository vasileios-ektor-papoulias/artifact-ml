from pathlib import Path

from artifact_core._libs.utils.system.package_importer import PackageImporter

from tests.base.dummy_artifact_toolkit.artifact_dependencies import (
    DummyArtifactResources,
    DummyResourceSpec,
)
from tests.base.dummy_artifact_toolkit.registries import (
    DummyArray,
    DummyArrayCollectionRegistry,
    DummyArrayCollectionType,
    DummyArrayRegistry,
    DummyPlot,
    DummyPlotCollectionRegistry,
    DummyPlotCollectionType,
    DummyPlotRegistry,
    DummyScoreCollectionRegistry,
    DummyScoreCollectionType,
    DummyScoreRegistry,
    DummyScoreType,
)

PackageImporter.import_all_from_package_path(path=Path(__file__).resolve().parent)
