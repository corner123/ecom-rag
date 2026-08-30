"""Physical parser helpers live behind :class:`DocumentRouter`'s fail-closed boundary.

The router deliberately owns dispatch and quarantine so parser failures cannot become
implicit plain text.  This module is the stable import home for future standalone parsers.
"""

from .pdf import pymupdf_extract

__all__ = ["pymupdf_extract"]
