"""spikesort: a from-scratch spike sorter with ground-truth benchmarking."""

from .evaluate import compare_to_ground_truth
from .pipeline import SortConfig, Sorting, sort
from .simulate import Recording, linear_probe_geometry, simulate_recording, tetrode_geometry

__all__ = [
    "Recording",
    "SortConfig",
    "Sorting",
    "compare_to_ground_truth",
    "linear_probe_geometry",
    "simulate_recording",
    "sort",
    "tetrode_geometry",
]

__version__ = "0.1.0"
