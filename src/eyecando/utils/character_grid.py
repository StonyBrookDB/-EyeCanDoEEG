"""The speller grid's own symbol alphabet.

A plain, dependency-free constant -- lives in utils rather than decode
(which imports it via `eyecando.decode.lm`'s re-export) so it can be
shared by anything that needs the grid layout without pulling in
decode's own LM/accumulator machinery.
"""

from __future__ import annotations

# Classic Farwell & Donchin (1988) 6x6 row-column speller grid, row-major --
# the standard layout from the original P300-speller literature this
# project's paradigm is built on. 26 letters + digits 1-9 + one space cell.
FARWELL_DONCHIN_GRID: list[str] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789 ")
