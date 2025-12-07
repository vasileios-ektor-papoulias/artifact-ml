import sys
from pathlib import Path
from typing import Callable, Generator, List

import pytest
from artifact_core._utils.system.module_importer import ModuleImporter
from pytest import FixtureRequest


@pytest.fixture
def patch_sys_path() -> Generator[None, None, None]:
    original_sys_path = sys.path.copy()
    original_modules = set(sys.modules.keys())
    yield
    sys.path[:] = original_sys_path
    modules_to_remove = set(sys.modules.keys()) - original_modules
    for module in modules_to_remove:
        del sys.modules[module]


@pytest.fixture
def pkg_simple(tmp_path: Path) -> Path:
    package_dir = tmp_path / "simple_pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").touch()
    sub = package_dir / "sub"
    sub.mkdir()
    (sub / "__init__.py").touch()
    return package_dir


@pytest.fixture
def pkg_nested(tmp_path: Path) -> Path:
    package_dir = tmp_path / "nested_pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").touch()
    level1 = package_dir / "level1"
    level1.mkdir()
    (level1 / "__init__.py").touch()
    level2 = level1 / "level2"
    level2.mkdir()
    (level2 / "__init__.py").touch()
    level3 = level2 / "level3"
    level3.mkdir()
    (level3 / "__init__.py").touch()
    return package_dir


@pytest.fixture
def pkg_with_modules(tmp_path: Path) -> Path:
    package_dir = tmp_path / "module_pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").touch()
    utils = package_dir / "utils"
    utils.mkdir()
    (utils / "__init__.py").touch()
    (utils / "helpers.py").write_text("# helper module\n")
    (utils / "tools.py").write_text("# tools module\n")
    core = package_dir / "core"
    core.mkdir()
    (core / "__init__.py").touch()
    (core / "main.py").write_text("# main module\n")
    return package_dir


@pytest.fixture
def pkg_dispatcher(request: FixtureRequest) -> Callable[[str], Path]:
    def _get_package(scenario_name: str) -> Path:
        fixture_name = f"pkg_{scenario_name}"
        return request.getfixturevalue(fixture_name)

    return _get_package


@pytest.mark.unit
@pytest.mark.parametrize(
    "scenario_name, relative_path, expected_modules",
    [
        (
            "simple",
            "sub",
            [],
        ),
        (
            "nested",
            "level1",
            [
                "nested_pkg.level1.level2",
                "nested_pkg.level1.level2.level3",
            ],
        ),
        (
            "nested",
            "level1/level2",
            [
                "nested_pkg.level1.level2.level3",
            ],
        ),
        (
            "nested",
            "level1/level2/level3",
            [],
        ),
        (
            "with_modules",
            "utils",
            [
                "module_pkg.utils.helpers",
                "module_pkg.utils.tools",
            ],
        ),
        (
            "with_modules",
            "core",
            [
                "module_pkg.core.main",
            ],
        ),
    ],
)
def test_import_modules(
    patch_sys_path: Generator[None, None, None],
    pkg_dispatcher: Callable[[str], Path],
    scenario_name: str,
    relative_path: str,
    expected_modules: List[str],
):
    pkg_root = pkg_dispatcher(scenario_name)
    path = pkg_root / relative_path
    ModuleImporter.import_modules(path=str(path), root=str(pkg_root))
    parent_dir = str(pkg_root.parent)
    assert parent_dir in sys.path
    for expected_module in expected_modules:
        assert expected_module in sys.modules
