"""
Data ingestion module
"""

from .loaders import StructuredDataLoader, UnstructuredDataLoader, DataIntegrator

__all__ = ["StructuredDataLoader", "UnstructuredDataLoader", "DataIntegrator"]
