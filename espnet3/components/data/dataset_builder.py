"""Base interface for dataset builders."""

from __future__ import annotations

from abc import ABC, abstractmethod


class DatasetBuilder(ABC):
    """Prepare dataset artifacts for a dataset module."""

    @abstractmethod
    def build(self, **kwargs) -> None:
        """Ensure dataset artifacts exist."""
