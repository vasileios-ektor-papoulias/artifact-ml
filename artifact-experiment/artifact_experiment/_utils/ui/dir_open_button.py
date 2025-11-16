from dataclasses import dataclass
from typing import Optional

from artifact_experiment._utils.system.dir_opener import DirectoryOpener
from IPython.display import display
from ipywidgets import Button, HBox, Layout


@dataclass
class DirectoryOpenButtonConfig:
    open_button_icon: str = "folder-open"
    open_button_width: str = "180px"
    open_button_height: str = "40px"
    open_button_color: str = "#4CAF50"
    open_button_font_weight: str = "600"
    open_button_font_size: str = "14px"
    path_label_font_size: str = "12px"
    path_label_font_style: str = "italic"


class DirectoryOpenButton:
    _default_description = "Open Dir"

    def __init__(
        self,
        path: str,
        description: Optional[str] = None,
        config: DirectoryOpenButtonConfig = DirectoryOpenButtonConfig(),
    ):
        if description is None:
            description = self._default_description
        self._path = path
        self._description = description
        self._config = config
        self._render_button()

    @property
    def button(self) -> Button:
        return self._button

    def click(self):
        self._on_click(None)

    def _render_button(self):
        self._button = self._create_open_button()
        path_label = self._create_path_label()
        display(HBox([self._button, path_label]))

    def _create_open_button(self) -> Button:
        cfg = self._config
        btn = Button(
            description=self._description,
            icon=cfg.open_button_icon,
            layout=Layout(width=cfg.open_button_width, height=cfg.open_button_height),
            style={
                "button_color": cfg.open_button_color,
                "font_weight": cfg.open_button_font_weight,
                "font_size": cfg.open_button_font_size,
            },
            tooltip=self._path,
        )
        btn.on_click(self._on_click)
        return btn

    def _create_path_label(self) -> Button:
        cfg = self._config
        return Button(
            description=self._path,
            disabled=True,
            layout=Layout(width="auto", height=cfg.open_button_height),
            style={
                "button_color": "white",
                "font_size": cfg.path_label_font_size,
                "font_style": cfg.path_label_font_style,
            },
        )

    def _on_click(self, _):
        DirectoryOpener.open_directory(path=self._path)
