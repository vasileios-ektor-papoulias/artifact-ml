from dataclasses import dataclass

from IPython.display import display
from ipywidgets import Button, HBox, Layout

from artifact_experiment.libs.utils.directory_opener import DirectoryOpener


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
    def __init__(
        self,
        path: str,
        description: str,
        config: DirectoryOpenButtonConfig = DirectoryOpenButtonConfig(),
    ):
        self._path = path
        self._description = description
        self._config = config
        self._opener = DirectoryOpener()
        self._render_button()

    def _render_button(self):
        open_btn = self._create_open_button()
        path_label = self._create_path_label()
        display(HBox([open_btn, path_label]))

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
        self._opener.open_directory(path=self._path)
