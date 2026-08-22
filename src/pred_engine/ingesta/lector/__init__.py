"""Adaptadores de E/S pasiva: CSV crudo hacia DataFrame."""

from pred_engine.ingesta.lector.extractor_csv import ExtractionArtifact, extract_csv
from pred_engine.ingesta.lector.hashing import hash_sha256_archivo

__all__ = [
    "ExtractionArtifact",
    "extract_csv",
    "hash_sha256_archivo",
]
