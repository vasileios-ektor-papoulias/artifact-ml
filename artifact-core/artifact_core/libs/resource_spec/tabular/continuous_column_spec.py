from typing import TypeVar

from artifact_core.libs.resource_spec.tabular.column_spec import ColumnSpec

ContinuousColumnSpecT = TypeVar("ContinuousColumnSpecT", bound="ContinuousColumnSpec")


class ContinuousColumnSpec(ColumnSpec):
    pass
