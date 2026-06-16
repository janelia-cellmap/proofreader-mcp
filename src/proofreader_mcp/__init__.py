"""proofreader-mcp: an MCP resource + state layer for image-centric proofreading
of segmentation objects.

The server organizes per-object composite mesh screenshots, exposes them as MCP
resources, and persists the host LLM's classifications, the derived rankings, and
human review verdicts. It stores no model and makes no inference calls.
"""

__version__ = "0.1.0"
