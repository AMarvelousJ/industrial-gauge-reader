"""Visual meter-style classification utilities.

The classifier deliberately consumes image pixels only.  Paths are used to
build labels during training and to identify rows in reports, never as model
features.
"""

from .manifest import ManifestAudit, ManifestEntry, parse_markdown_manifest

__all__ = ["ManifestAudit", "ManifestEntry", "parse_markdown_manifest"]

