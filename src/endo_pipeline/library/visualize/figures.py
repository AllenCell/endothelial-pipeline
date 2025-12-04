import xml.etree.ElementTree as ET
from typing import NamedTuple
from pathlib import Path

INCHES_TO_PIXELS = 96


class FigurePanel(NamedTuple):
    """Configuration for figure panel."""

    letter: str
    """Panel letter"""

    path: str
    """Path to the plot as SVG."""

    x_position: float
    """Horizontal panel position in inches (left is 0)."""

    y_position: float
    """Vertical panel position in inches (top is 0)."""

    x_offset: float
    """Horizontal offset of plot from panel position in inches."""

    y_offset: float
    """Vertical offset of plot from panel position in inches."""


def build_empty_figure(width: float, height: float) -> ET.Element:

    # Convert inches to points.
    width = int(width * INCHES_TO_PIXELS)
    height = int(height * INCHES_TO_PIXELS)

    # Register SVG namespaces.
    ET.register_namespace('',"http://www.w3.org/2000/svg")
    ET.register_namespace('xlink',"http://www.w3.org/1999/xlink")

    # Create empty figure of given size.
    figure = ET.fromstring(f'<svg width="{width}px" height="{height}px"></svg>')

    # Add white background to figure.
    ET.SubElement(figure, "rect", {
        "width": f"{width}px",
        "height": f"{height}px",
        "fill": "white"
    })

    return figure

def build_panel_group(root: ET.Element, x: float, y: float) -> ET.Element:
    x = x * INCHES_TO_PIXELS
    y = y * INCHES_TO_PIXELS

    return ET.SubElement(root, "g",{"transform": f"translate({x},{y})"})


def add_panel_letter(root: ET.Element, letter: str) -> None:
    element = ET.SubElement(root, "text", {
        "font-size": "14px",
        "x": "7",
        "y": "15",
        "font-family": "Arial",
        "font-weight": "bold",
        "text-anchor": "middle",
    })
    element.text = letter


def build_figure_from_panels(figure_panels: list[FigurePanel], output_path: Path, width: float, height: float) -> None:
    figure = build_empty_figure(width, height)

    for panel in figure_panels:
        group = build_panel_group(figure, panel.x_position, panel.y_position)
        offset = build_panel_group(group, panel.x_offset, panel.y_offset)
        offset.extend(ET.parse(panel.path).getroot())
        add_panel_letter(group, panel.letter)

    ET.indent(figure, space="    ", level=0)
    output_path.write_text(ET.tostring(figure, encoding="unicode"))
