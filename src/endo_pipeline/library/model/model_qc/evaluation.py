"""Core model evaluation logic for model QC."""

from typing import NamedTuple


class ModelKey(NamedTuple):
    """Manifest + run name pair uniquely identifying a model in an evaluation run.

    Hashable so it can serve as a dict key, and provides a ``.label`` property
    for display on figures (two-line ``manifest`` / ``run`` format used in
    suptitles and tick labels).
    """

    manifest_name: str
    run_name: str

    @property
    def label(self) -> str:
        """Human-readable label for display on figures."""
        return f"{self.manifest_name}\n{self.run_name}"
