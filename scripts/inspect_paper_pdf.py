import hashlib
import json
from pathlib import Path
import pymupdf
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
source = root / "docs/publication/SignalDNA_Research_Paper_DRAFT.pdf"
qa = root / "docs/publication/qa"
pages = qa / "pages"
pages.mkdir(parents=True, exist_ok=True)

assert source.is_file(), "Academic PDF missing"
records = []
thumbs = []
problems = []

with pymupdf.open(source) as pdf:
    for index, page in enumerate(pdf):
        number = index + 1
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        image.save(pages / f"page_{number:02d}.png")

        thumb = image.copy()
        thumb.thumbnail((700, 920))
        thumbs.append(thumb)

        blocks = [
            b for b in page.get_text("blocks")
            if len(b) > 6 and b[6] == 0 and b[4].strip()
        ]
        text = page.get_text()
        images = len(page.get_image_info())
        escaped = [
            [round(v, 1) for v in b[:4]]
            for b in blocks
            if b[0] < 20 or b[1] < 15
            or b[2] > page.rect.width - 20
            or b[3] > page.rect.height - 15
        ]

        if len(text.strip()) < 80:
            problems.append(f"Page {number}: insufficient extractable text")
        if escaped:
            problems.append(f"Page {number}: text near or beyond page boundary")

        records.append({
            "page": number,
            "characters": len(text),
            "embedded_images": images,
            "boundary_flags": escaped,
        })
        print(f"PAGE {number}: {len(text)} characters, {images} images, {len(escaped)} boundary flags")

    assert len(pdf) >= 4, "Unexpectedly short paper"
    assert sum(r["embedded_images"] for r in records) >= 3, "Figures missing"

columns = 2
rows = (len(thumbs) + 1) // 2
width, height = 1480, rows * 960 + 30
sheet = Image.new("RGB", (width, height), "#E9EDF2")
draw = ImageDraw.Draw(sheet)

for i, thumb in enumerate(thumbs):
    x = (i % columns) * 740 + (740 - thumb.width) // 2
    y = (i // columns) * 960 + 35
    sheet.paste(thumb, (x, y))
    draw.text((x, y - 22), f"PAGE {i+1}", fill="#24364D")

contact = qa / "academic_paper_contact_sheet.png"
sheet.save(contact, optimize=True)

report = {
    "source": str(source.relative_to(root)),
    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "pages": records,
    "issues": problems,
}
(qa / "academic_paper_preflight.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

print("CONTACT SHEET:", contact)
print("QA REPORT:", qa / "academic_paper_preflight.json")
print("ISSUES:", len(problems))
for issue in problems:
    print("WARNING:", issue)
if problems:
    raise SystemExit("SIGNALDNA-PDF-VISUAL-QA-241: REVIEW REQUIRED")
print("SIGNALDNA-PDF-VISUAL-QA-241: PASS")
