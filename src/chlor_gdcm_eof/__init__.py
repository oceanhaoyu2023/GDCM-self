"""GDCM-EOF chlorophyll-a reconstruction package."""

from .config import InferenceConfig

__all__ = ["GDCMEOFGenerator", "InferenceConfig", "generator"]


def __getattr__(name: str):
    if name in {"GDCMEOFGenerator", "generator"}:
        from .model import GDCMEOFGenerator, generator

        return {"GDCMEOFGenerator": GDCMEOFGenerator, "generator": generator}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
