from typing import List, Optional

from artifact_core.libs.resource_spec.labels.spec import LabelsSpec


class BinaryLabelsSpec(LabelsSpec):
    _binary_cardinality = 2

    def add_label(self, label_name: str, classes: Optional[List[str]] = None) -> None:
        """
        Add a new label (task). Requires exactly two class values.
        """
        classes = classes or []
        norm = list(dict.fromkeys(classes))
        if len(norm) != self._binary_cardinality:
            raise ValueError(
                f"BinaryLabelsSpec requires exactly 2 classes per label. "
                f"Got {len(norm)} for '{label_name}': {norm!r}"
            )
        super().add_label(label_name=label_name, classes=norm)

    def set_label_classes(self, label_name: str, classes: List[str]) -> None:
        """
        Replace the allowed classes for an existing label. Requires exactly two.
        """
        norm = list(dict.fromkeys(classes))
        if len(norm) != self._binary_cardinality:
            raise ValueError(
                f"BinaryLabelsSpec requires exactly 2 classes per label. "
                f"Got {len(norm)} for '{label_name}': {norm!r}"
            )
        super().set_label_classes(label_name=label_name, classes=norm)

    def _validate_internal(self) -> None:
        """
        Extend base validation with a binary constraint on every label.
        """
        # Run the base validations: alignment, de-duping, id_name check, etc.
        super()._validate_internal()

        # Enforce binary cardinality for each label.
        bad: List[str] = []
        for ln, spec in self._dict_labels.items():
            classes = list(dict.fromkeys(spec.get("classes", [])))
            # Write back normalized classes to keep the internal spec tidy.
            self._dict_labels[ln]["classes"] = classes
            if len(classes) != self._binary_cardinality:
                bad.append(f"{ln}: {classes!r} (n={len(classes)})")

        if bad:
            details = "\n  ".join(bad)
            raise ValueError(
                "BinaryLabelsSpec violation — each label must have exactly 2 classes:\n  " + details
            )
