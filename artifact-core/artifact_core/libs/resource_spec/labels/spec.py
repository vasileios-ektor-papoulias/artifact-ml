import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import pandas as pd

from artifact_core.libs.resource_spec.labels.protocol import LabelsSpecProtocol

LabelsSpecT = TypeVar("LabelsSpecT", bound="LabelsSpec")


class LabelsSpec(LabelsSpecProtocol):
    """
    Static metadata for a multi-label classification schema.

    _internal_spec = {
        "id_name": str,
        "labels_order": List[str],
        "labels": {
            "<label_name>": {"classes": List[str]},
            ...
        },
    }
    """

    def __init__(self, internal_spec: Dict[str, Any]):
        self._internal_spec = internal_spec
        self._validate_internal()

    @classmethod
    def build(
        cls: Type[LabelsSpecT],
        id_name: str,
        ls_labels: Optional[List[str]] = None,
        label_classes_map: Optional[Dict[str, List[str]]] = None,
    ) -> LabelsSpecT:
        if ls_labels is None:
            ls_labels = []

        if label_classes_map is None:
            label_classes_map = {ln: [] for ln in ls_labels}
        else:
            # Ensure union alignment between provided names and map keys
            for ln in ls_labels:
                label_classes_map.setdefault(ln, [])
            for ln in list(label_classes_map.keys()):
                if ln not in ls_labels:
                    ls_labels.append(ln)

        internal_spec = cls._build_internal_spec(
            id_name=id_name,
            ls_labels=ls_labels,
            label_classes_map=label_classes_map,
        )
        return cls(internal_spec=internal_spec)

    @classmethod
    def from_df(
        cls: Type[LabelsSpecT],
        df: pd.DataFrame,
        id_col: str,
        label_names: Optional[List[str]] = None,
        coerce_to_str: bool = True,
        dropna_classes: bool = True,
    ) -> LabelsSpecT:
        if id_col not in df.columns:
            raise ValueError(f"'{id_col}' not found in DataFrame columns.")

        if label_names is None:
            label_names = [c for c in df.columns if c != id_col]

        label_classes_map: Dict[str, List[str]] = {}
        for ln in label_names:
            if ln not in df.columns:
                raise ValueError(f"Label column '{ln}' not found in DataFrame.")
            series = df[ln]
            uniques = (
                series.dropna().unique().tolist() if dropna_classes else series.unique().tolist()
            )
            if coerce_to_str:
                uniques = [str(v) for v in uniques if pd.notna(v)]
            uniq_ordered = list(dict.fromkeys(uniques))  # stable unique
            label_classes_map[ln] = sorted(uniq_ordered)

        return cls.build(
            id_name=id_col,
            ls_labels=label_names,
            label_classes_map=label_classes_map,
        )

    @property
    def id_name(self) -> str:
        return self._internal_spec["id_name"]

    @property
    def ls_labels(self) -> List[str]:
        return list(self._internal_spec["labels_order"])

    @property
    def n_labels(self) -> int:
        return len(self.ls_labels)

    @property
    def label_classes_map(self) -> Dict[str, List[str]]:
        return {k: list(v["classes"]) for k, v in self._dict_labels.items()}

    @property
    def label_classes_count_map(self) -> Dict[str, int]:
        return {k: len(v["classes"]) for k, v in self._dict_labels.items()}

    @property
    def _id_name(self) -> str:
        return self._internal_spec["id_name"]

    @property
    def _dict_labels(self) -> Dict[str, Dict[str, List[str]]]:
        return self._internal_spec["labels"]

    @property
    def _ls_labels(self) -> List[str]:
        return self._internal_spec["labels_order"]

    @_ls_labels.setter
    def _ls_labels(self, labels_order: List[str]) -> None:
        self._internal_spec["labels_order"] = labels_order

    def get_classes(self, label_name: str) -> List[str]:
        ls_classes = self._get_classes(label_name=label_name)
        return ls_classes

    def get_n_classes(self, label_name: str) -> int:
        ls_classes = self._get_classes(label_name=label_name)
        return len(ls_classes)

    def add_label(self, label_name: str, classes: Optional[List[str]] = None) -> None:
        if label_name in self._dict_labels:
            raise ValueError(f"Label '{label_name}' already exists.")
        self._dict_labels[label_name] = {"classes": list(dict.fromkeys(classes or []))}
        self._ls_labels.append(label_name)

    def set_label_classes(self, label_name: str, classes: List[str]) -> None:
        self._require_label(label_name)
        self._dict_labels[label_name]["classes"] = list(dict.fromkeys(classes))

    def remove_label(self, label_name: str) -> None:
        self._require_label(label_name)
        del self._dict_labels[label_name]
        self._ls_labels = [ln for ln in self._ls_labels if ln != label_name]

    def serialize(self) -> str:
        return json.dumps(self._internal_spec)

    @classmethod
    def deserialize(cls: Type[LabelsSpecT], json_str: str) -> LabelsSpecT:
        return cls(internal_spec=json.loads(json_str))

    def export(self, filepath: Path) -> None:
        if filepath.suffix != ".json":
            filepath = filepath.with_suffix(".json")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.serialize())

    @classmethod
    def load(cls: Type[LabelsSpecT], filepath: Path) -> LabelsSpecT:
        if filepath.suffix != ".json":
            filepath = filepath.with_suffix(".json")
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.deserialize(f.read())

    def _get_classes(self, label_name: str) -> List[str]:
        self._require_label(label_name)
        return list(self._dict_labels[label_name]["classes"])

    def _require_label(self, label_name: str) -> None:
        if label_name not in self._dict_labels:
            raise ValueError(
                f"Unknown label '{label_name}'. Known labels: {sorted(self._dict_labels.keys())}"
            )

    def _validate_internal(self) -> None:
        names_in_map = set(self._dict_labels.keys())
        names_in_order = set(self._ls_labels)
        for extra in names_in_map - names_in_order:
            self._ls_labels.append(extra)
        self._ls_labels = [ln for ln in self._ls_labels if ln in names_in_map]
        for ln, spec in self._dict_labels.items():
            classes = spec.get("classes", [])
            spec["classes"] = list(dict.fromkeys(classes))
        idn = self._internal_spec.get("id_name")
        if not isinstance(idn, str) or not idn:
            raise ValueError("id_name must be a non-empty string.")

    @classmethod
    def _build_internal_spec(
        cls,
        id_name: str,
        ls_labels: List[str],
        label_classes_map: Dict[str, List[str]],
    ) -> Dict[str, object]:
        labels_block: Dict[str, Dict[str, List[str]]] = {
            ln: {"classes": list(dict.fromkeys(label_classes_map.get(ln, [])))} for ln in ls_labels
        }
        internal_spec: Dict[str, object] = {
            "id_name": id_name,
            "labels_order": list(ls_labels),
            "labels": labels_block,
        }
        return internal_spec
