"""Dataset audit, conversion, filtering, and split utilities."""

from src.dataset.audit import run_audit
from src.dataset.convert import convert_dataset, voc_xml_to_yolo_lines
from src.dataset.duplicates import find_near_duplicates
from src.dataset.splits import create_leakage_safe_splits
from src.dataset.validate import validate_processed_dataset
from src.dataset.week3_layout import create_week3_splits

__all__ = [
    "run_audit",
    "convert_dataset",
    "voc_xml_to_yolo_lines",
    "find_near_duplicates",
    "create_leakage_safe_splits",
    "validate_processed_dataset",
    "create_week3_splits",
]
