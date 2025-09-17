from typing import List, Optional

from artifact_core.libs.resources.labels.store import LabelStore


class BinaryLabelStore(LabelStore):
    """
    Specialization of LabelStore that enforces *binary* label spaces.

    Guarantees:
      - Every label has exactly 2 (unique) class values.
      - Stored example values are already validated by the parent implementation
        to be within the allowed classes.
    """

    _BINARY_CARDINALITY = 2

    # --------------------------- mutation overrides ---------------------------

    def add_label_name(self, label_name: str, classes: Optional[List[str]] = None) -> None:
        """
        Register a new label (task) with its class space.
        Requires exactly two unique class values.
        """
        norm = list(dict.fromkeys(classes or []))
        if len(norm) != self._BINARY_CARDINALITY:
            raise ValueError(
                f"BinaryLabelStore requires exactly 2 classes per label. "
                f"Got {len(norm)} for '{label_name}': {norm!r}"
            )
        super().add_label_name(label_name=label_name, classes=norm)

    # ------------------------------- validation -------------------------------

    def _validate_internal(self) -> None:
        """
        Extend base validation to enforce binary cardinality for each label.
        """
        # Run base validations: align keys, de-dup values, verify store consistency.
        super()._validate_internal()

        # Enforce exactly two classes per label.
        bad = []
        for ln, classes in self._label_classes.items():
            uniq = list(dict.fromkeys(classes))
            # Write back normalized classes to keep internal state tidy.
            self._label_classes[ln] = uniq
            if len(uniq) != self._BINARY_CARDINALITY:
                bad.append(f"{ln}: {uniq!r} (n={len(uniq)})")

        if bad:
            details = "\n  ".join(bad)
            raise ValueError(
                "BinaryLabelStore violation — each label must have exactly 2 classes:\n  " + details
            )
