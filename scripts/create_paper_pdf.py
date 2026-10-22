import re
from html import escape
from pathlib import Path
import pymupdf
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table, TableStyle, KeepTogether, Spacer, HRFlowable, PageBreak

import matplotlib
dejavu_fonts = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
root = Path(__file__).resolve().parents[1]
src = root / 'docs/RESEARCH_PAPER.md'
figdir = root / 'docs/figures'
out = root / 'docs/publication/SignalDNA_Research_Paper_DRAFT.pdf'
out.parent.mkdir(parents=True, exist_ok=True)
for name in ('learning_accuracy', 'learning_loss', 'cpu_cuda_epoch_benchmark'):
    assert (figdir / (name + '.png')).is_file(), name + ': missing graph'

def font(name, paths, fallback):
    for path in paths:
        if Path(path).is_file():
            pdfmetrics.registerFont(TTFont(name, path))
            return name
    return fallback

body = font('GeorgiaPaper', [str(dejavu_fonts / 'DejaVuSerif.ttf')], 'Times-Roman')
bodybold = font('GeorgiaPaperBold', [str(dejavu_fonts / 'DejaVuSerif-Bold.ttf')], 'Times-Bold')
head = font('SegoePaper', [str(dejavu_fonts / 'DejaVuSans.ttf')], 'Helvetica')
headbold = font('SegoePaperBold', [str(dejavu_fonts / 'DejaVuSans-Bold.ttf')], 'Helvetica-Bold')
pdfmetrics.registerFontFamily(body, normal=body, bold=bodybold)
navy, blue, gray = (colors.HexColor(x) for x in ('#162944', '#23648D', '#52627A'))

def style(name, family, size, leading, color=colors.black, **extra):
    return ParagraphStyle(name, fontName=family, fontSize=size, leading=leading, textColor=color, **extra)

styles = {
 'title': style('Title', headbold, 18.5, 23, navy, spaceAfter=14),
 'meta': style('Meta', head, 9.4, 13, gray, spaceAfter=12),
 'h1': style('H1', headbold, 12.2, 16, navy, spaceBefore=12, spaceAfter=6, keepWithNext=True),
 'h2': style('H2', headbold, 10.3, 14, blue, spaceBefore=11, spaceAfter=5, keepWithNext=True),
 'p': style('Body', body, 9.3, 14.2, colors.HexColor('#222A33'), spaceAfter=5.0, splitLongWords=True),
 'abstract': style('Abstract', body, 9.3, 14.5, navy, leftIndent=11, rightIndent=11, spaceAfter=9),
 'ref': style('Reference', body, 8.3, 12.4, gray, leftIndent=18, firstLineIndent=-18, spaceAfter=6, splitLongWords=True),
 'caption': style('Caption', head, 8.2, 11.7, gray, spaceBefore=6, spaceAfter=8),
 'cell': style('Cell', body, 9, 13),
 'th': style('Th', headbold, 9, 13, colors.white),
 'eq': style('Equation', 'Courier', 9, 13.5),
}

def inline(s):
    s = escape(s)
    s = re.sub(r'`([^`]+)`', r'<font face="Courier">\1</font>', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'(https?://[^\s<>]+)', r'<link href="\1" color="#23648D">\1</link>', s)
    return s

def figure(n, image, caption):
    path = figdir / image
    with PILImage.open(path) as pic:
        w, h = pic.size
    draw_width = min(410, 300 * w / h)
    return KeepTogether([Image(str(path), width=draw_width, height=draw_width * h/w),
                         Paragraph('<b>Figure %d.</b> %s' % (n, escape(caption)), styles['caption'])])

captions = {
 '3.1 Final held-out classification': [
    (1, 'learning_accuracy.png', 'Training and validation accuracy across the archived 60-epoch NumPy run.'),
    (2, 'learning_loss.png', 'Training and validation binary cross-entropy; divergence illustrates the generalization gap.')],
 '3.3 CPU/GPU performance': [
    (3, 'cpu_cuda_epoch_benchmark.png', 'Five warmed-up epoch measurements per implementation; bars show medians and diamonds individual runs. Float64, batch size 64; workload-specific observations.')],
}

def pageframe(canvas, doc):
    canvas.saveState()
    width, height = letter
    canvas.setStrokeColor(colors.HexColor('#CCD5DF'))
    canvas.line(55, height-39, width-55, height-39)
    canvas.line(55, 42, width-55, 42)
    canvas.setFont(head, 8)
    canvas.setFillColor(gray)
    canvas.drawString(56, height-32, 'SIGNALDNA  /  INDEPENDENT RESEARCH')
    canvas.drawString(56, 29, 'Research manuscript  |  Not peer reviewed')
    canvas.drawRightString(width-56, 29, str(doc.page))
    canvas.restoreState()

lines = src.read_text(encoding='utf-8-sig').splitlines()
assert lines and lines[0].startswith('# '), 'Missing original manuscript title'
title = lines[0][2:].strip()
story = [Paragraph(escape(title), styles['title']),
         Paragraph('<b>David Boring</b>  |  Independent research  |  October 2026', styles['meta']),
         HRFlowable(width='100%', thickness=1.8, color=blue, spaceAfter=10)]
mode, section, inserted = '', '', set()

def add_figures():
    if section in captions and section not in inserted:
        story.extend(figure(*args) for args in captions[section])
        inserted.add(section)

i = 1
while i < len(lines):
    s = lines[i]
    if not s.strip():
        i += 1
        continue
    if s.startswith(('## ', '### ')):
        add_figures()
        sub = s.startswith('### ')
        section = s[4 if sub else 3:].strip()
        if not sub:
            mode = 'abstract' if section == 'Abstract' else 'ref' if section == 'References' else ''
        if section == 'References': story.append(PageBreak())
        story.append(Paragraph(escape(section.upper() if mode == 'abstract' and not sub else section), styles['h2' if sub else 'h1']))
        i += 1
        continue
    if s.startswith('|'):
        rowtext = []
        while i < len(lines) and lines[i].startswith('|'):
            rowtext.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
            i += 1
        rowtext = [r for r in rowtext if not all(re.fullmatch(r':?-{3,}:?', c) for c in r)]
        assert all(len(r)==len(rowtext[0]) for r in rowtext), 'Malformed table'
        if len(rowtext[0]) == 2 and rowtext[0][0] == 'Metric':
            for row in rowtext[1:]:
                if row[0] in ('Accuracy', 'Precision', 'Recall'):
                    row[1] = f'{float(row[1])*100:.2f}%'
                elif row[0] in ('F1', 'ROC-AUC', 'Binary cross-entropy'):
                    row[1] = f'{float(row[1]):.4f}'
        cells = [[Paragraph(inline(c), styles['th' if idx == 0 else 'cell']) for c in row] for idx, row in enumerate(rowtext)]
        table = Table(cells, colWidths=[258, 216] if len(cells[0])==2 else None, hAlign='CENTER', repeatRows=1)
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F4F7FA')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('ALIGN',(1,1),(-1,-1),'RIGHT')]))
        story.extend((table, Spacer(1,9)))
        continue
    if s.startswith('    '):
        eq = []
        while i < len(lines) and lines[i].startswith('    '):
            eq.append(Paragraph(escape(lines[i].strip()),styles['eq']))
            i += 1
        table=Table([[x] for x in eq],colWidths=[472],hAlign='CENTER')
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#F1F5F9')),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10)]))
        story.extend((table,Spacer(1,9)))
        continue
    para=[]
    while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','    ')):
        para.append(lines[i].strip())
        i+=1
    if not para:
        raise ValueError('Unsupported source syntax at line %d' % (i+1))
    t=' '.join(para)
    story.append(Paragraph(inline(t),styles['meta' if t.startswith('**Status:**') else mode if mode else 'p']))

add_figures()
assert inserted==set(captions), 'Not all scientific figures inserted'
doc=SimpleDocTemplate(str(out),pagesize=letter,leftMargin=60,rightMargin=60,topMargin=57,bottomMargin=53,title=title,author='David Boring',subject='Independent research, not peer reviewed')
doc.build(story,onFirstPage=pageframe,onLaterPages=pageframe)

with pymupdf.open(out) as pdf:
    extracted='\n'.join(p.get_text() for p in pdf)
    assert 4<=len(pdf)<=18 and all(k in extracted for k in ('ABSTRACT','References','78.78','Figure 1.','Figure 2.','Figure 3.'))
    assert sum(len(p.get_images()) for p in pdf)>=3, 'Missing graph images'
    print('OUTPUT:',out,'| PAGES:',len(pdf),'| BYTES:',out.stat().st_size)
print('SIGNALDNA-ACADEMIC-PDF-DRAFT-240: PASS')
