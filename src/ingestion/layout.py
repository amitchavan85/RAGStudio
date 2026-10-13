"""
Layout-aware text extraction (roadmap P6: ingestion quality).

1. Columns: many reports print text in two columns. Reading a page line by line across the
   full width interleaves them ("...recover to t ariffs and the continued..."). We look for a
   vertical gutter that prose lines on both sides never cross, then read each two-column band
   left column first, right column second. Lines that cross the gutter (titles, full-width
   paragraphs, headers) are read across the full width, in page order.
2. Running headers/footers ("GLOBAL ECONOMIC PROSPECTS | JUNE 2026 CHAPTER 1 19") repeat on
   most pages and pollute chunks; lines at the top/bottom of a page that repeat (ignoring
   numbers) on at least 30% of the pages of a document are removed.
3. Drop capitals: a large first letter is extracted apart from its word ("G rowth"); a lone
   capital at the start of a line followed by a lowercase word is re-joined.
4. OCR: pages without a text layer (scans) are rendered to an image and read with RapidOCR
   (`pip install rapidocr-onnxruntime`), if installed and OCR is enabled.
"""
from __future__ import annotations

import logging
import re
import statistics
from collections import Counter
from functools import lru_cache
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_LINE_TOL = 3.0  # words whose tops differ by less than this (points) are on the same line


# ---------- columns ----------

def _lines(words: list) -> List[list]:
    lines: List[list] = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if lines and abs(lines[-1][0]["top"] - w["top"]) < _LINE_TOL:
            lines[-1].append(w)
        else:
            lines.append([w])
    return lines


def _crosses(line: list, x: float) -> bool:
    return any(w["x0"] < x < w["x1"] for w in line)


def find_gutter(words: list, width: float) -> Optional[float]:
    """x position of a two-column gutter, or None for single-column pages.

    Two columns means: almost no line crosses the gutter, both sides hold a good share of the
    words, and both sides are prose (several words per line), so a table with a gap in the
    middle is not mistaken for columns. Column lines need not line up across the gutter.
    """
    if len(words) < 40:
        return None
    lines = _lines(words)
    best: Optional[Tuple[int, float]] = None
    for x in [width * f / 200 for f in range(70, 131)]:  # 35%..65% of the width, half-percent steps
        crossing = sum(_crosses(line, x) for line in lines)
        if best is None or crossing < best[0]:
            best = (crossing, x)
    crossing, x = best
    if crossing > 0.15 * len(lines):
        return None
    left_words = sum(w["x1"] <= x for w in words)
    right_words = sum(w["x0"] >= x for w in words)
    if min(left_words, right_words) < 0.2 * len(words):
        return None
    left = [n for n in (sum(w["x1"] <= x for w in line) for line in lines) if n]
    right = [n for n in (sum(w["x0"] >= x for w in line) for line in lines) if n]
    if statistics.median(left) < 4 or statistics.median(right) < 4:
        return None  # a table, not prose columns
    return x


def column_aware_text(page) -> str:
    """Page text in reading order: full-width bands as-is, two-column bands left then right."""
    words = page.extract_words()
    gutter = find_gutter(words, page.width)
    if gutter is None:
        return (page.extract_text() or "").strip()

    lines = _lines(words)
    margin = 0.08 * page.height  # running headers/footers: always read full width
    bands: List[Tuple[bool, float, float]] = []  # (is_split, top, bottom)
    for line in lines:
        in_margin = line[0]["top"] < margin or line[0]["bottom"] > page.height - margin
        is_split = not in_margin and not _crosses(line, gutter)
        top = min(w["top"] for w in line)
        bottom = max(w["bottom"] for w in line)
        if bands and bands[-1][0] == is_split:
            bands[-1] = (is_split, bands[-1][1], max(bands[-1][2], bottom))
        else:
            bands.append((is_split, top, bottom))

    parts: List[str] = []
    for is_split, top, bottom in bands:
        y0, y1 = max(0, top - 1), min(page.height, bottom + 1)
        boxes = [(0, y0, gutter, y1), (gutter, y0, page.width, y1)] if is_split else [(0, y0, page.width, y1)]
        for box in boxes:
            text = (page.crop(box).extract_text() or "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


# ---------- clean-up ----------

_DROP_CAP_RE = re.compile(r"(?m)^([B-HJ-Z]) ([a-z]{2,})")  # not A/I: "A dog", "I am" are real words


def fix_drop_caps(text: str) -> str:
    return _DROP_CAP_RE.sub(r"\1\2", text)


def _signature(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def _edge_indexes(lines: List[str], edge_lines: int) -> List[int]:
    """Indexes of the first/last non-blank lines; short pages only use their very first and last."""
    nonblank = [i for i, l in enumerate(lines) if l.strip()]
    n = edge_lines if len(nonblank) > 2 * edge_lines + 2 else 1
    return sorted(set(nonblank[:n] + nonblank[-n:]))


def remove_running_headers(pages: List[str], edge_lines: int = 2, min_share: float = 0.3) -> List[str]:
    """Drop header/footer lines that repeat (ignoring digits) on >= min_share of the pages.

    0.3, not 0.5: books alternate headers on left and right pages, so each version appears
    on only about half of the pages. Only lines with words qualify, so numeric table rows
    ("2019 21,430 +2.3%" looks like "# #,# +#.#%" on every page) are never removed.
    """
    if len(pages) < 4:
        return pages
    counts: Counter = Counter()
    for text in pages:
        lines = text.splitlines()
        counts.update({_signature(lines[i]) for i in _edge_indexes(lines, edge_lines)})
    repeated = {sig for sig, n in counts.items()
                if n >= max(3, min_share * len(pages)) and len(re.findall(r"[a-z]{3,}", sig)) >= 2}
    if not repeated:
        return pages
    out = []
    for text in pages:
        lines = text.splitlines()
        edges = set(_edge_indexes(lines, edge_lines))
        out.append("\n".join(l for i, l in enumerate(lines) if not (i in edges and _signature(l) in repeated)))
    return out


# ---------- OCR ----------

@lru_cache(maxsize=1)
def _ocr_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return None
    return RapidOCR()


def ocr_available() -> bool:
    return _ocr_engine() is not None


def ocr_page(page, resolution: int = 200) -> str:
    """Render the page and read it with RapidOCR; lines in top-to-bottom, left-to-right order."""
    import numpy as np

    engine = _ocr_engine()
    if engine is None:
        return ""
    image = page.to_image(resolution=resolution).original.convert("RGB")
    result, _ = engine(np.array(image))
    if not result:
        return ""
    boxes = sorted(((min(p[1] for p in box), min(p[0] for p in box), text) for box, text, _score in result))
    lines: List[List[tuple]] = []
    for top, left, text in boxes:
        if lines and abs(lines[-1][0][0] - top) < 10:
            lines[-1].append((top, left, text))
        else:
            lines.append([(top, left, text)])
    return "\n".join(" ".join(t for _, _, t in sorted(line, key=lambda b: b[1])) for line in lines)
