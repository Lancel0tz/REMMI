"""Sandbox fallback for `memqa.retrieve.utils.RetrievalItem`.

The real definition lives in `memqa.retrieve.utils`, but importing that
module pulls in `memqa.retrieve.retrievers`, which depends on `torch`. On
a CPU-only sandbox (or in unit tests we want to keep cheap), torch may not
be installed.

This shim mirrors only the fields the hybrid retriever actually reads:
``item_id``, ``modality``, ``text``, ``metadata``. It is *only* used as a
fallback — production callers will get the real class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RetrievalItem:
    item_id: str
    modality: str
    text: str
    image_path: Optional[Path] = None
    video_path: Optional[Path] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
