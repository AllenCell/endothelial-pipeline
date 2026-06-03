import inspect
import xml.etree.ElementTree as ET
from functools import wraps
from pathlib import Path
from textwrap import shorten, wrap
from typing import NamedTuple

from endo_pipeline.io import get_output_path, slugify

INCHES_TO_PIXELS = 96

ILLUSTRATOR_SCALING_FACTOR = 0.75
"""Scaling factor to rescale figure dimensions for use in Adobe Illustrator."""


class FigurePanel(NamedTuple):
    """Configuration for figure panel."""

    letter: str
    """Panel letter"""

    path: Path
    """Path to the plot as SVG."""

    x_position: float
    """Horizontal panel position in inches (left is 0)."""

    y_position: float
    """Vertical panel position in inches (top is 0)."""

    x_offset: float
    """Horizontal offset of plot from panel position in inches."""

    y_offset: float
    """Vertical offset of plot from panel position in inches."""


def figure_panel(description: str):
    """Decorator for figure panels that adds support for creating placeholders."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, placeholder: bool, **kwargs):
            if placeholder:
                # Try to identify an output path from the method arguments.
                # First, check for "output_path" in the keyword arguments. Then,
                # check for any instance of Path in the arguments. Then, as a
                # fallback, create a new output path.
                kwarg_output_path = kwargs.get("output_path", None)
                arg_output_path = next((arg for arg in args if isinstance(arg, Path)), None)
                output_path = kwarg_output_path or arg_output_path or get_output_path("placeholder")

                # Try to identify figure size from method arguments. First,
                # check for "figure_size" argument in keyword arguments. Then,
                # check for a default "figure_size" argument in the signature.
                # Finally, as a fallback, use figure size of 2 x 2.
                kwarg_figure_size = kwargs.get("figure_size", None)
                param = inspect.signature(func).parameters.get("figure_size", None)
                default_figure_size = (
                    param.default
                    if param is not None and param.default is not inspect.Parameter.empty
                    else None
                )
                figure_size = kwarg_figure_size or default_figure_size or (2, 2)

                return build_empty_panel(output_path, description, *figure_size)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def parse_placeholder_panels(
    include_panels: list[str] | None, all_panels: list[str]
) -> dict[str, dict[str, bool]]:
    """Parse included panels to placeholder indicator."""

    # Default to all panels
    if include_panels is None:
        include_panels = all_panels

    # Filter out invalid panels and convert to upper case
    include_panels = [panel.upper() for panel in include_panels if panel.upper() in all_panels]

    # Set placeholder to True for panel that are not included, False otherwise
    return {panel: {"placeholder": panel not in include_panels} for panel in all_panels}


def build_empty_panel(output_path: Path, description: str, width: float, height: float) -> Path:
    """Build empty placeholder panel with description text."""

    # Convert inches to points.
    width = int(width * INCHES_TO_PIXELS * ILLUSTRATOR_SCALING_FACTOR)
    height = int(height * INCHES_TO_PIXELS * ILLUSTRATOR_SCALING_FACTOR)

    # Register SVG namespaces.
    ET.register_namespace("", "http://www.w3.org/2000/svg")

    # Create empty panel of given size.
    panel = ET.fromstring(
        f'<svg width="{width}px" height="{height}px" xmlns="http://www.w3.org/2000/svg"></svg>'
    )

    # Add gray background
    ET.SubElement(
        panel,
        "rect",
        {
            "width": f"{width}px",
            "height": f"{height}px",
            "fill": "#000",
            "fill-opacity": "0.2",
            "stroke": "#999",
        },
    )

    # Add panel description
    font_size = 14
    characters_per_line = round(width / font_size / 0.6)
    wrap_text = wrap(description, characters_per_line)
    panel_text = ET.SubElement(
        panel,
        "g",
        {
            "transform": f"translate({width//2},{height//2 + font_size//4})",
            "fill": "#999",
            "font-size": f"{font_size}px",
            "font-family": "Arial",
            "text-anchor": "middle",
        },
    )
    for index, text in enumerate(wrap_text):
        offset = (index - len(wrap_text) / 2 + 0.5) * font_size
        ET.SubElement(panel_text, "text", {"y": f"{offset}"}).text = text

    # Write panel to path.
    ET.indent(panel, space="    ", level=0)
    slug = slugify(str(width), "x", str(height), shorten(description, 80))
    output_file = output_path / f"placeholder_{slug}.svg"
    output_file.write_text(ET.tostring(panel, encoding="unicode"), encoding="utf-8")

    return output_file


def build_empty_figure(width: float, height: float) -> ET.Element:

    # Convert inches to points.
    width = int(width * INCHES_TO_PIXELS * ILLUSTRATOR_SCALING_FACTOR)
    height = int(height * INCHES_TO_PIXELS * ILLUSTRATOR_SCALING_FACTOR)

    # Register SVG namespaces.
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

    # Create empty figure of given size.
    figure = ET.fromstring(f'<svg width="{width}px" height="{height}px"></svg>')

    # Add white background to figure.
    ET.SubElement(figure, "rect", {"width": f"{width}px", "height": f"{height}px", "fill": "white"})

    return figure


def build_panel_group(root: ET.Element, x: float, y: float) -> ET.Element:
    x = x * INCHES_TO_PIXELS * ILLUSTRATOR_SCALING_FACTOR
    y = y * INCHES_TO_PIXELS * ILLUSTRATOR_SCALING_FACTOR

    return ET.SubElement(root, "g", {"transform": f"translate({x},{y})"})


def add_panel_letter(root: ET.Element, letter: str) -> None:
    element = ET.SubElement(
        root,
        "text",
        {
            "font-size": "14px",
            "x": "7",
            "y": "15",
            "font-family": "Arial",
            "font-weight": "bold",
            "text-anchor": "middle",
        },
    )
    element.text = letter


def build_figure_from_panels(
    figure_panels: list[FigurePanel], output_path: Path, width: float, height: float
) -> None:

    figure = build_empty_figure(width, height)

    for panel in figure_panels:
        group = build_panel_group(figure, panel.x_position, panel.y_position)
        offset = build_panel_group(group, panel.x_offset, panel.y_offset)
        offset.extend(ET.parse(panel.path).getroot())
        add_panel_letter(group, panel.letter)

    ET.indent(figure, space="    ", level=0)
    output_path.write_text(ET.tostring(figure, encoding="unicode"), encoding="utf-8")
