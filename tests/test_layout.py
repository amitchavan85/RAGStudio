"""Roadmap P6: ingestion quality (two-column reading order, running headers, drop caps, OCR)."""
import os

import pdfplumber
import pytest
from helpers import PDF_DIR

from src.ingestion.layout import find_gutter, fix_drop_caps, ocr_available, remove_running_headers
from src.ingestion.loader import load_pdf

LEFT = [f"Left sentence {i} on inflation." for i in range(30)]
RIGHT = [f"Right sentence {i} on trade." for i in range(30)]
TOPICS = ["energy", "housing", "credit", "tourism", "farming", "mining"]  # body text differs per page


def _two_column_pdf(path, header="ANNUAL REPORT 2026 | PAGE {n}", pages=1):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    for n in range(1, pages + 1):
        c.setFont("Helvetica", 9)
        c.drawString(72, height - 40, header.format(n=n))  # running header (full width)
        c.setFont("Helvetica", 9)
        for i in range(30):
            # Columns deliberately not aligned: right column starts 5 points lower.
            topic = TOPICS[(n - 1) % len(TOPICS)] if n > 1 else ""
            c.drawString(72, height - 80 - i * 14, f"{topic} {LEFT[i]}"[:48].strip())
            c.drawString(320, height - 85 - i * 14, f"{topic} {RIGHT[i]}"[:48].strip())
        c.showPage()
    c.save()


def test_two_columns_are_read_left_then_right(tmp_path):
    pdf = tmp_path / "two_col.pdf"
    _two_column_pdf(pdf)
    text = load_pdf(str(pdf))[0].text
    assert text.index("Left sentence 29 ") < text.index("Right sentence 0 ")
    for line in text.splitlines():  # no line mixes the two columns
        assert not ("Left sentence" in line and "Right sentence" in line), line


def test_running_headers_removed_across_pages(tmp_path):
    pdf = tmp_path / "report.pdf"
    _two_column_pdf(pdf, pages=5)
    pages = load_pdf(str(pdf))
    assert len(pages) == 5
    assert not any("ANNUAL REPORT" in p.text for p in pages)
    assert all("Left sentence 0 " in p.text for p in pages)  # body kept on every page


def test_single_column_and_table_pages_are_not_split():
    """The GDP samples are single-column prose plus tables: none may be treated as two columns."""
    for name in sorted(os.listdir(PDF_DIR)):
        with pdfplumber.open(os.path.join(PDF_DIR, name)) as pdf:
            for page in pdf.pages:
                assert find_gutter(page.extract_words(), page.width) is None, (name, page.page_number)


def test_gdp_table_rows_survive_extraction():
    text = load_pdf(os.path.join(PDF_DIR, "usa_gdp_report.pdf"))[0].text
    assert "2023 27,360 +2.5%" in text


def test_remove_running_headers_unit():
    pages = [f"WORLD OUTLOOK | CHAPTER 1 {n}\nBody text {n}.\nMore body.\nPage {n} footer" for n in range(1, 7)]
    cleaned = remove_running_headers(pages)
    assert not any("WORLD OUTLOOK" in p or "footer" in p for p in cleaned)
    assert all(f"Body text {n}." in p for n, p in zip(range(1, 7), cleaned))
    short = pages[:3]
    assert remove_running_headers(short) == short  # too few pages to tell what repeats


@pytest.mark.parametrize("raw, fixed", [
    ("G rowth in the euro area", "Growth in the euro area"),
    ("W eather-related disasters", "Weather-related disasters"),
    ("A dog barked.", "A dog barked."),          # real one-letter words stay
    ("I am here.", "I am here."),
    ("Vitamin C is good", "Vitamin C is good"),  # only at the start of a line
])
def test_fix_drop_caps(raw, fixed):
    assert fix_drop_caps(raw) == fixed


@pytest.mark.skipif(not ocr_available(), reason="rapidocr-onnxruntime not installed")
def test_scanned_page_is_read_with_ocr(tmp_path):
    """Render a GDP page to an image-only PDF (a 'scan'), then extract it: OCR must recover the facts."""
    with pdfplumber.open(os.path.join(PDF_DIR, "usa_gdp_report.pdf")) as pdf:
        image = pdf.pages[0].to_image(resolution=150).original.convert("RGB")
    scan = tmp_path / "scan.pdf"
    image.save(scan, "PDF", resolution=150)
    with pdfplumber.open(scan) as pdf:
        assert not pdf.pages[0].chars  # truly no text layer

    docs = load_pdf(str(scan))
    assert docs and docs[0].metadata.get("ocr") is True
    text = docs[0].text
    for fact in ("Washington", "27,360", "Services"):
        assert fact in text, fact


def test_ocr_can_be_disabled(tmp_path, monkeypatch):
    from src.config import get_settings

    with pdfplumber.open(os.path.join(PDF_DIR, "usa_gdp_report.pdf")) as pdf:
        image = pdf.pages[0].to_image(resolution=72).original.convert("RGB")
    scan = tmp_path / "scan.pdf"
    image.save(scan, "PDF", resolution=72)
    monkeypatch.setenv("OCR", "off")
    get_settings.cache_clear()
    try:
        assert load_pdf(str(scan)) == []  # no text layer and OCR off -> page skipped
    finally:
        get_settings.cache_clear()
