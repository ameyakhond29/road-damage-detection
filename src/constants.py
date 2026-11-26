"""Project-wide constants for RDD2022 road damage detection."""

from __future__ import annotations

# Default reproducibility seed for the entire project.
DEFAULT_SEED = 6845

# CRDDC / RDD2022 four main damage classes.
CLASS_TO_ID: dict[str, int] = {
    "D00": 0,  # Longitudinal crack
    "D10": 1,  # Transverse crack
    "D20": 2,  # Alligator crack
    "D40": 3,  # Pothole
}

ID_TO_CLASS: dict[int, str] = {v: k for k, v in CLASS_TO_ID.items()}
CLASS_NAMES: list[str] = ["D00", "D10", "D20", "D40"]

# Labels that appear in VOC XMLs but are out of scope for this project.
UNSUPPORTED_LABELS: set[str] = {
    "D50",
    "Repair",
    "D01",
    "D11",
    "D43",
    "D44",
    "Block crack",
    "Other Corruption",
}

# Aerial subset with a different camera setup — excluded from training data.
EXCLUDE_COUNTRIES: set[str] = {"China_Drone"}

# Known country folders in RDD2022.
KNOWN_COUNTRIES: tuple[str, ...] = (
    "China_Drone",
    "China_MotorBike",
    "Czech",
    "India",
    "Japan",
    "Norway",
    "United_States",
)

# Consecutive-filename block size used for leakage-safe grouping.
GROUP_BLOCK_SIZE = 50

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

DATASET_VERSION = "RDD2022_released_through_CRDDC2022"
OFFICIAL_ARCHIVE_URL = (
    "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/RDD2022.zip"
)
# Mirror linked from https://github.com/sekilab/RoadDamageDetector (Figshare 21431547).
OFFICIAL_MIRROR_URL = "https://ndownloader.figshare.com/files/38030910"
OFFICIAL_MIRROR_SIZE_BYTES = 13_257_587_040
ARCHIVE_NAMES: tuple[str, ...] = ("RDD2022.zip", "RDD2022_released_through_CRDDC2022.zip")
RDD2022_ROOT_ENV = "RDD2022_ROOT"

DOWNLOAD_INSTRUCTIONS = """
RDD2022 dataset not found (or incomplete).

Download and extract the official dataset with:

  python scripts/download_rdd2022.py

or point RDD2022_ROOT (or --data-root) at an existing RDD2022 folder.
Sources: https://github.com/sekilab/RoadDamageDetector
  (S3 archive, Figshare mirror https://doi.org/10.6084/m9.figshare.21431547)

Extract so the layout looks like:

  data/raw/RDD2022/
  ├── China_Drone/train/images/ ... annotations/xmls/
  ├── China_MotorBike/train/images/ ... annotations/xmls/
  ├── Czech/train/...
  ├── India/train/...
  ├── Japan/train/...
  ├── Norway/train/...
  └── United_States/train/...

For unit tests / debugging only, pass --dataset fixture (synthetic data).
""".strip()
