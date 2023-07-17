from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from typing import Iterable, Union


bt_gray = colors.Color(
    name="bt_gray",
    c50="#e5e5e5",
    c100="#d3d3d3",
    c200="#c1c1c1",
    c300="#afafaf",
    c400="#9e9e9e",
    c500="#8c8c8c",
    c600="#7a7a7a",
    c700="#686868",
    c800="#565656",
    c900="#444444",
    c950="#333333",
)

class BrefingToolTheme(Base):
    def __init__(
        self,
        *,
        primary_hue: Union[colors.Color, str] = colors.orange,
        secondary_hue: Union[colors.Color, str]  = bt_gray,
        neutral_hue: Union[colors.Color, str]  = bt_gray,
        text_size: Union[sizes.Size, str] = sizes.text_md,
        spacing_size: Union[sizes.Size, str] = sizes.spacing_md,
        radius_size:Union[sizes.Size, str] = sizes.radius_md,
        font: Union[fonts.Font,
        str,
        Iterable[Union[fonts.Font, str]]] = (
            fonts.GoogleFont("Source Sans Pro"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ),
        font_mono: Union[fonts.Font,
        str,
        Iterable[Union[fonts.Font, str]]] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "Consolas",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        orange ='#ff7402'
        light_orange = '#ff811a'
        super().set(
            body_background_fill='#393939',
            block_background_fill='#2f2f2f',
            button_secondary_background_fill_hover=orange,
            button_secondary_text_color_hover='white',
            button_primary_background_fill=orange,
            button_primary_background_fill_hover=light_orange,
            button_primary_text_color='white',
        )