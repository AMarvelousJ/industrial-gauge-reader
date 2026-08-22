"""Auditable analog-gauge reading baseline (no model training)."""

from .geometry import analyze_pointer, clockwise_angle_degrees
from .dial_geometry import DialGeometry, estimate_dial_geometry

__all__ = ["DialGeometry", "analyze_pointer", "clockwise_angle_degrees", "estimate_dial_geometry"]
