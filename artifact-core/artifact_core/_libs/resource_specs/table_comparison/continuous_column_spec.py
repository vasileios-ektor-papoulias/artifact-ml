from typing import TypeVar

from artifact_core._libs.resource_specs.table_comparison.column_spec import ColumnSpec

ContinuousColumnSpecT = TypeVar("ContinuousColumnSpecT", bound="ContinuousColumnSpec")


class ContinuousColumnSpec(ColumnSpec):
    pass
