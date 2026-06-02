"""Laser Job Time Estimation Tool.

Batch-processes DXF files with color-based speed assignments,
generates G-code, and estimates cut times via a Duet 2 controller
or offline calculation.
"""

from laser_time_tool.cli import estimate_folder

__all__ = ["estimate_folder"]
