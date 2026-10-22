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

import matplotlib
fontdir = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
pdfmetrics.registerFont(TTFont("BriefRegular", str(fontdir / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("BriefBold", str(fontdir / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("BriefGeorgia", str(fontdir / "DejaVuSerif.ttf")))

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
    text(44, 28, "SignalDNA  |  Independent research  |  Apache-2.0", 7.2, muted)
    text(447, 28, "1 / 1  |  Not peer reviewed", 7.0, muted)
header("ONE-PAGE RESEARCH BRIEF")
text(44, 713, "SignalDNA", 28, navy, True)
text(44, 688, "Neural networks from first principles | Synthetic transient detection", 9.4, muted)
for x,value,label in [(44,f"{accuracy*100:.2f}%","Held-out accuracy"),(222,f"{auc:.4f}","Held-out ROC-AUC"),(400,"67,713","Parameters")]:
    panel(x,610,168,60)
    text(x+12,638,value,18,navy,True)
    text(x+12,620,label,7.6,muted)
paragraph("Original NumPy network: 1024 inputs, 64 and 32 ReLU units, 1 sigmoid output. Trained on balanced synthetic transients; 12,000 signals, 1,800 held out. Motivation: learning AI before exploring embedded and FPGA applications.",44,598,524,size=8.2,leading=11.4,limit=48)
panel(44,338,254,205)
panel(310,338,258,205)
text(55,522,"TRAINING & GENERALIZATION",8.5,blue,True)
text(321,522,"CPU VS CUDA",8.5,blue,True)
image_fit("learning_accuracy.png",54,365,234,145)
image_fit("cpu_cuda_epoch_benchmark.png",320,365,238,145)
text(54,351,"60 epochs; train vs. validation",7.2,muted)
text(320,351,"Five warmed-up runs; float64, batch 64",7.2,muted)
text(44,317,f"Train {train*100:.2f}%  |  Validation {validation*100:.2f}%  |  Held-out {accuracy*100:.2f}%",9,navy,True)
text(44,299,f"Median epoch: CPU {cpu:.2f} ms  |  CUDA {gpu:.2f} ms  |  GPU/CPU {ratio:.2f}x",8.8,navy,True)
panel(44,178,524,105)
text(56,265,"WHAT THE EXPERIMENTS SHOW",9,blue,True)
paragraph(f"Training accuracy greatly exceeded validation accuracy: generalization remains the central challenge. CPU and CUDA metrics matched at 61 archived checkpoints (maximum absolute difference {difference:.2e}); the CPU was faster for this specific small workload. CUDA timings include synchronization and CPU transfers.",56,251,500,size=8.3,leading=12,limit=72)
panel(44,62,524,105)
text(56,149,"LIMITATIONS & NEXT STEPS",9,orange,True)
paragraph("Synthetic signals only; one principal architecture and seed; one locked held-out evaluation. No validated classical DSP baseline, real-sensor testing, or FPGA deployment. Next: classical baselines, regularization, multiple seeds, real data, and hardware inference measurements.",56,135,500,size=8.3,leading=12,limit=72)
footer(1)
c.save()
qa = pub / "qa"
qa.mkdir(exist_ok=True)
with pymupdf.open(out) as pdf:
    assert len(pdf) == 1, "Brief must be exactly one page"
    images = sum(len(page.get_image_info()) for page in pdf)
    assert images >= 2, "Missing research charts"
    content = "\n".join(p.get_text() for p in pdf)
    assert "78.78%" in content and "WHAT THE EXPERIMENTS SHOW" in content
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
print("PAGES: 1 | EMBEDDED FIGURES:", images)
print("CONTACT SHEET:", qa / "project_brief_contact_sheet.png")
print("SIGNALDNA-PROJECT-BRIEF-244: PASS")
