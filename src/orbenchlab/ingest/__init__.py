"""Convert upstream runner outputs into small, report-safe result slices."""

from .harbor import HarborIngestResult, ingest_harbor_bundle

__all__ = ["HarborIngestResult", "ingest_harbor_bundle"]

