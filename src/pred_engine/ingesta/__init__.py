"""Ingestion and characterisation layer.

Reads raw transaction files (CSV/Parquet), validates them, builds quality
logs, constructs per-SKU time series, and computes demand-profile features
that downstream layers consume.
"""
