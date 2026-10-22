import json, re, statistics
from html import escape
from pathlib import Path
import pymupdf
from PIL import Image as PILImage
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

root = Path(__file__).resolve().parents[1]
pub = root / "docs/publication"
figs = root / "docs/figures"
pub.mkdir(parents=True, exist_ok=True)
out = pub / "SignalDNA_Project_Brief_DRAFT.pdf"

manuscript = (root / "docs/RESEARCH_PAPER.md").read_text(encoding="utf-8-sig")
curve = json.loads((root / "results/numpy_cpu_60_epochs_logged.json").read_text(encoding="utf-8"))
benchmark = json.loads((root / "results/cpu_gpu_benchmark_v1.json").read_text(encoding="utf-8"))
comparison = json.loads((root / "results/cpu_gpu_curve_comparison.json").read_text(encoding="utf-8"))

def metric(name):
    match = re.search(r'^\|\s*' + re.escape(name) + r'\s*\|\s*([0-9.]+)\s*\|', manuscript, re.M)
    assert match, f"Missing manuscript metric: {name}"
    return float(match.group(1))

accuracy = metric("Accuracy")
auc = metric("ROC-AUC")
train = float(curve["epochs"][-1]["training_accuracy"])
validation = float(curve["epochs"][-1]["validation_accuracy"])
assert len(curve["epochs"]) == 61
assert abs(accuracy - 0.7877777777777778) < 1e-10
assert "1024-64-32-1" in manuscript
assert (root / "LICENSE").is_file()

cpu = statistics.median(benchmark["measurements_seconds"]["numpy_cpu"]) * 1000
gpu = statistics.median(benchmark["measurements_seconds"]["pytorch_cuda"]) * 1000
ratio = gpu / cpu
assert abs(ratio - float(benchmark["median_gpu_over_cpu_time_ratio"])) < 0.02
assert int(comparison["epochs_compared"]) == 61
assert str(comparison["result"]).upper() == "PASS"

difference = max(float(v) for v in comparison["maximum_absolute_differences"].values())
assert difference < 1e-12

fontdir = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("BriefRegular", str(fontdir / "segoeui.ttf")))
pdfmetrics.registerFont(TTFont("BriefBold", str(fontdir / "segoeuib.ttf")))
pdfmetrics.registerFont(TTFont("BriefGeorgia", str(fontdir / "georgia.ttf")))

navy = HexColor("#132842")
blue = HexColor("#23699B")
orange = HexColor("#CF742B")
light = HexColor("#F2F6FA")
muted = HexColor("#526379")
white = HexColor("#FFFFFF")
ink = HexColor("#263445")

c = canvas.Canvas(str(out), pagesize=letter)
c.setTitle("SignalDNA | Independent Engineering Research Brief")
c.setAuthor("David Boring")
c.setSubject("Synthetic transient detection and CPU/GPU neural-network characterization")

def text(x, y, value, size=10, color=ink, bold=False):
    c.setFont("BriefBold" if bold else "BriefRegular", size)
    c.setFillColor(color)
    c.drawString(x, y, str(value))

def paragraph(value, x, top, width, size=9.2, leading=13.5, limit=None, color=ink):
    style = ParagraphStyle("brief", fontName="BriefRegular", fontSize=size,
                           leading=leading, textColor=color)
    item = Paragraph(escape(value), style)
    _, h = item.wrap(width, 500)
    if limit is not None:
        assert h <= limit, f"Text overflow: {value[:50]} ({h:.1f}>{limit})"
    item.drawOn(c, x, top-h)
    return h

def panel(x, y, w, h, background=light):
    c.setFillColor(background)
    c.roundRect(x, y, w, h, 10, fill=1, stroke=0)

def image_fit(name, x, y, w, h):
    p = figs / name
    assert p.is_file(), f"Missing archived-data figure: {p}"
    with PILImage.open(p) as im:
        iw, ih = im.size
    scale = min(w/iw, h/ih)
    dw, dh = iw*scale, ih*scale
    c.drawImage(str(p), x+(w-dw)/2, y+(h-dh)/2,
                width=dw, height=dh, mask="auto")

def header(section):
    c.setFillColor(navy)
    c.rect(0, 772, 612, 20, fill=1, stroke=0)
    text(44, 750, "VERGE.io  /  SIGNALDNA", 9, blue, True)
    text(448, 750, section, 8, muted)

def footer(page):
    c.setStrokeColor(HexColor("#D9E1E9"))
    c.line(44, 43, 568, 43)
    text(44, 28, "David Boring  |  October 2026  |  Software: Apache-2.0", 7.5, muted)
    text(462, 28, f"{page} / 2  |  Not peer reviewed", 7.2, muted)

def metric_card(x, value, label, color=navy):
    panel(x, 593, 168, 72)
    text(x+13, 626, value, 20, color, True)
    text(x+13, 606, label, 8, muted)

# PAGE ONE — PROJECT AND CLASSIFICATION
header("RESEARCH OVERVIEW")
text(44, 708, "SignalDNA", 29, navy, True)
text(44, 683, "Learning neural networks from first principles", 12, muted)

metric_card(44, f"{accuracy*100:.2f}%", "Held-out accuracy")
metric_card(222, f"{auc:.4f}", "Held-out ROC-AUC", blue)
metric_card(400, "67,713", "Network parameters", orange)

text(44, 569, "THE EXPERIMENT", 10, blue, True)
paragraph(
    "Built to learn neural networks from first principles and explore "
    "future FPGA and embedded engineering applications, SignalDNA "
    "uses a NumPy reference network to classify synthetic transients. "
    "It examines generalization, numerical equivalence and CPU/GPU performance.",
    44, 556, 524, limit=52
)

panel(44, 431, 524, 65)
text(58, 478, "NETWORK ARCHITECTURE", 9, blue, True)
text(58, 455, "1024 inputs   >   64 ReLU   >   32 ReLU   >   1 sigmoid",
     12, navy, True)
text(58, 440, "12,000 balanced signals  |  1,800 held-out  |  2,048 Hz  |  0.5 s  |  60 epochs",
     8.2, muted)

panel(44, 104, 524, 315)
text(60, 396, "01  /  TRAINING AND GENERALIZATION", 10, navy, True)
image_fit("learning_accuracy.png", 60, 132, 492, 252)
text(60, 116, "Figure 1. Archived training and validation accuracy, epochs 0-60.",
     7.8, muted)

text(44, 85,
     f"Training {train*100:.2f}%  /  Validation {validation*100:.2f}%  /  Test {accuracy*100:.2f}%",
     9.2, navy, True)
text(44, 68, "A substantial generalization gap remains despite high training accuracy.",
     8.4, muted)
footer(1)
c.showPage()

# PAGE TWO — PERFORMANCE AND ENGINEERING IMPLICATIONS
header("PERFORMANCE & NEXT STEPS")
text(44, 708, "What the experiments revealed", 21, navy, True)
text(44, 683, "Reproducibility, unexpected outcomes, and engineering lessons",
     10.2, muted)

panel(44, 374, 524, 282)
text(60, 631, "02  /  CPU VERSUS CUDA", 10, navy, True)
image_fit("cpu_cuda_epoch_benchmark.png", 61, 400, 490, 218)
text(60, 385, "Figure 2. Five warmed-up runs; float64 and batch size 64.",
     7.8, muted)

for x, value, label in [
    (44, f"{cpu:.2f} ms", "CPU median epoch"),
    (222, f"{gpu:.2f} ms", "CUDA median epoch"),
    (400, f"{ratio:.3f}x", "CUDA / CPU time"),
]:
    panel(x, 306, 168, 57)
    text(x+12, 328, value, 15, navy, True)
    text(x+12, 313, label, 8, muted)

panel(44, 183, 254, 112)
panel(310, 183, 258, 112)
text(58, 272, "NUMERICAL EQUIVALENCE", 9.2, blue, True)
paragraph(
    f"CPU and CUDA learning metrics matched across 61 archived "
    f"checkpoints. Maximum absolute metric difference: {difference:.3e}. "
    "Both harnesses shared NumPy metric evaluation.",
    58, 258, 226, size=8.8, leading=12.3, limit=72
)
text(324, 272, "WHAT WE LEARNED", 9.2, blue, True)
paragraph(
    "GPU acceleration is not automatically faster. "
    "This specific small-workload benchmark favored CPU execution. "
    "The finding does not establish general CPU superiority.",
    324, 258, 225, size=8.8, leading=12.3, limit=72
)

panel(44, 60, 254, 112)
panel(310, 60, 258, 112)
text(58, 149, "RESEARCH LIMITATIONS", 9.2, orange, True)
paragraph(
    "Synthetic data only; one principal architecture and seed; "
    "a single locked held-out evaluation. No real-sensor, "
    "embedded, or FPGA deployment has been demonstrated.",
    58, 135, 226, size=8.8, leading=12.3, limit=70
)
text(324, 149, "NEXT RESEARCH DIRECTIONS", 9.2, blue, True)
paragraph(
    "Classical signal baselines, regularization, real sensor signals, "
    "cross-seed analysis, quantization, FPGA inference, "
    "latency, resource use, and power measurements.",
    324, 135, 225, size=8.8, leading=12.3, limit=70
)

footer(2)
c.save()

qa = pub / "qa"
qa.mkdir(exist_ok=True)
with pymupdf.open(out) as pdf:
    assert len(pdf) == 2, "Brief must be exactly two pages"
    images = sum(len(page.get_image_info()) for page in pdf)
    assert images >= 2, "Missing research charts"
    content = "\n".join(p.get_text() for p in pdf)
    assert "78.78%" in content and "NUMERICAL EQUIVALENCE" in content
    previews = []
    for index, page in enumerate(pdf):
        pix = page.get_pixmap(matrix=pymupdf.Matrix(1.25, 1.25), alpha=False)
        preview = PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
        preview.save(qa / f"brief_page_{index+1}.png")
        previews.append(preview)
    sheet = PILImage.new("RGB",
                         (sum(im.width for im in previews), max(im.height for im in previews)),
                         "#E9EDF2")
    offset = 0
    for preview in previews:
        sheet.paste(preview, (offset, 0))
        offset += preview.width
    sheet.save(qa / "project_brief_contact_sheet.png")
print("OUTPUT:", out)
print("PAGES: 2 | EMBEDDED FIGURES:", images)
print("CONTACT SHEET:", qa / "project_brief_contact_sheet.png")
print("SIGNALDNA-PROJECT-BRIEF-244: PASS")
