"""Pascal VOC XML parsing helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VocBox:
    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class VocAnnotation:
    path: Path
    width: int
    height: int
    boxes: list[VocBox] = field(default_factory=list)
    parse_error: str | None = None
    malformed_objects: list[str] = field(default_factory=list)


def parse_voc_xml(xml_path: Path) -> VocAnnotation:
    """Parse a Pascal VOC XML file into boxes and image size."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as exc:
        return VocAnnotation(
            path=xml_path, width=0, height=0, boxes=[], parse_error=str(exc)
        )

    size = root.find("size")
    width = height = 0
    if size is not None:
        try:
            width = int(float((size.findtext("width") or "0").strip()))
            height = int(float((size.findtext("height") or "0").strip()))
        except ValueError:
            width = height = 0

    boxes: list[VocBox] = []
    malformed: list[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bnd = obj.find("bndbox")
        if not name or bnd is None:
            malformed.append(name or "<no name>")
            continue
        try:
            xmin = float((bnd.findtext("xmin") or "0").strip())
            ymin = float((bnd.findtext("ymin") or "0").strip())
            xmax = float((bnd.findtext("xmax") or "0").strip())
            ymax = float((bnd.findtext("ymax") or "0").strip())
        except ValueError:
            malformed.append(name)
            continue
        boxes.append(VocBox(label=name, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))

    return VocAnnotation(
        path=xml_path, width=width, height=height, boxes=boxes, malformed_objects=malformed
    )
