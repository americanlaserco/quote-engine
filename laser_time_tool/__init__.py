"""Laser Job Time Estimation Tool.

Batch-processes DXF files with color-based speed assignments,
generates G-code, and estimates cut times offline.
"""

# Lazy CLI import — only available when the cli module is present
# (it's not bundled for the public Render deployme