import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Type, TypeVar

import numpy as np
import pandas as pd

LabelStoreT = TypeVar("LabelStoreT", bound="LabelStore")


class LabelStore:
    def __init__(
        self,
        labels_store: Optional[Dict[str, Dict[str, str]]] = None,
        label_classes: Optional[Dict[str, List[str]]] = None,
        id_name: str = "id",
    ):
        self._labels_store: Dict[str, Dict[str, str]] = labels_store or {}
        self._label_classes: Dict[str, List[str]] = label_classes or {}
        self._id_name: str = id_name
        self._validate_internal()

    @classmethod
    def build(
        cls: Type[LabelStoreT],
        label_names: Optional[List[str]] = None,
        label_classes: Optional[Dict[str, List[str]]] = None,
        id_name: str = "id",
    ) -> LabelStoreT:
        if label_names is None:
            label_names = []

        if label_classes is None:
            label_classes = {ln: [] for ln in label_names}
        else:
            for ln in label_names:
                label_classes.setdefault(ln, [])
            for ln in list(label_classes.keys()):
                if ln not in label_names:
                    label_names.append(ln)
        labels_store = {ln: {} for ln in label_classes.keys()}
        return cls(labels_store=labels_store, label_classes=label_classes, id_name=id_name)

    @classmethod
    def from_df(
        cls: Type[LabelStoreT],
        df: pd.DataFrame,
        id_col: str,
        label_names: Optional[List[str]] = None,
        coerce_to_str: bool = True,
        dropna_classes: bool = True,
    ) -> LabelStoreT:
        if id_col not in df.columns:
            raise ValueError(f"'{id_col}' not found in DataFrame columns.")
        if label_names is None:
            label_names = [c for c in df.columns if c != id_col]
        label_classes: Dict[str, List[str]] = {}
        labels_store: Dict[str, Dict[str, str]] = {ln: {} for ln in label_names}
        for ln in label_names:
            if ln not in df.columns:
                raise ValueError(f"Label column '{ln}' not found in DataFrame.")
            series = df[ln]
            if dropna_classes:
                uniques = series.dropna().unique().tolist()
            else:
                uniques = series.unique().tolist()
            if coerce_to_str:
                uniques = [str(v) for v in uniques if pd.notna(v)]
            label_classes[ln] = sorted(list(dict.fromkeys(uniques)))
        for _, row in df.iterrows():
            ex_id = row[id_col]
            if pd.isna(ex_id):
                continue
            ex_id = str(ex_id)
            for ln in label_names:
                val = row[ln]
                if pd.isna(val):
                    continue
                if coerce_to_str:
                    val = str(val)
                labels_store[ln][ex_id] = val

        return cls(labels_store=labels_store, label_classes=label_classes, id_name=id_col)

    @property
    def label_names(self) -> List[str]:
        return list(self._label_classes.keys())

    @property
    def id_name(self) -> str:
        return self._id_name

    @property
    def label_classes(self) -> Dict[str, List[str]]:
        return {k: v.copy() for k, v in self._label_classes.items()}

    @property
    def labels_store(self) -> Dict[str, Dict[str, str]]:
        return {k: v.copy() for k, v in self._labels_store.items()}

    @property
    def example_ids(self) -> List[str]:
        ids: Set[str] = set()
        for ln in self._labels_store:
            ids.update(self._labels_store[ln].keys())
        return sorted(ids)

    @property
    def df(self) -> pd.DataFrame:
        ids = self.example_ids
        if not ids and self.label_names:
            return pd.DataFrame(columns=[self._id_name] + self.label_names)
        data = {self._id_name: ids}
        for ln in self.label_names:
            col = []
            store = self._labels_store.get(ln, {})
            for ex_id in ids:
                col.append(store.get(ex_id, np.nan))
            data[ln] = col
        return pd.DataFrame(data)

    def add_label_name(self, label_name: str, classes: Optional[List[str]] = None) -> None:
        """
        Register a new label name (task) with its class space.
        """
        if label_name in self._label_classes:
            raise ValueError(f"Label '{label_name}' already exists.")
        self._label_classes[label_name] = list(dict.fromkeys(classes or []))
        self._labels_store[label_name] = {}

    def set_example_label(self, example_id: str, label_name: str, class_value: str) -> None:
        """
        Add or update the label value for an example under a specific label name.
        Validates that class_value is in the allowed class list for that label.
        """
        example_id = str(example_id)
        self._require_label_name(label_name)
        self._require_class_value(label_name, class_value)
        self._labels_store[label_name][example_id] = class_value

    def delete_examples(self, example_ids: Iterable[str]) -> None:
        """
        Remove any labels associated with the given example ids across all label names.
        """
        for raw_id in example_ids:
            ex_id = str(raw_id)
            for ln in self._labels_store.keys():
                self._labels_store[ln].pop(ex_id, None)

    def delete_example(self, example_id: str) -> None:
        self.delete_examples([example_id])

    def serialize(self) -> str:
        """
        Serialize to JSON string.
        """
        payload = {
            "id_name": self._id_name,
            "label_classes": self._label_classes,
            "labels_store": self._labels_store,
        }
        return json.dumps(payload)

    @classmethod
    def deserialize(cls: Type[LabelStoreT], json_str: str) -> LabelStoreT:
        payload = json.loads(json_str)
        return cls(
            labels_store=payload.get("labels_store", {}),
            label_classes=payload.get("label_classes", {}),
            id_name=payload.get("id_name", "id"),
        )

    def export(self, filepath: Path) -> None:
        """
        Save to a JSON file (adds .json suffix if missing).
        """
        if filepath.suffix != ".json":
            filepath = filepath.with_suffix(".json")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.serialize())

    @classmethod
    def load(cls: Type[LabelStoreT], filepath: Path) -> LabelStoreT:
        if filepath.suffix != ".json":
            filepath = filepath.with_suffix(".json")
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.deserialize(f.read())

    def _validate_internal(self) -> None:
        ls1 = set(self._labels_store.keys())
        ls2 = set(self._label_classes.keys())
        if ls1 != ls2:
            missing_in_store = ls2 - ls1
            missing_in_classes = ls1 - ls2
            if missing_in_store:
                for ln in missing_in_store:
                    self._labels_store[ln] = {}
            if missing_in_classes:
                for ln in missing_in_classes:
                    self._label_classes[ln] = []
        for ln in self.label_names:
            self._validate_label_values(ln)

    def _validate_label_values(self, label_name: str) -> None:
        classes = set(self._label_classes.get(label_name, []))
        if not classes:
            return
        store = self._labels_store.get(label_name, {})
        bad = {ex_id: val for ex_id, val in store.items() if val not in classes}
        if bad:
            example_list = ", ".join([f"{k}={v}" for k, v in list(bad.items())[:5]])
            raise ValueError(
                f"Found values not in declared classes for '{label_name}': {example_list}"
                + (" ..." if len(bad) > 5 else "")
            )

    def _require_label_name(self, label_name: str) -> None:
        if label_name not in self._label_classes:
            raise ValueError(
                f"Unknown label '{label_name}'. Known labels: {sorted(self._label_classes.keys())}"
            )

    def _require_class_value(self, label_name: str, class_value: str) -> None:
        classes = self._label_classes.get(label_name, [])
        if classes and class_value not in classes:
            raise ValueError(
                f"Class value '{class_value}' not allowed for label '{label_name}'. "
                f"Allowed: {classes}"
            )
