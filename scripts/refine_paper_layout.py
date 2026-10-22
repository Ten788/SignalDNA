from pathlib import Path

p = Path("scripts/create_paper_pdf.py")
s = p.read_text(encoding="utf-8-sig")

changes = {
    "draw_width = min(450, 300 * w / h)": "draw_width = min(410, 300 * w / h)",
    "spaceAfter=6.5, splitLongWords=True": "spaceAfter=5.0, splitLongWords=True",
    "spaceBefore=15, spaceAfter=7": "spaceBefore=12, spaceAfter=6",
    "spaceBefore=13, spaceAfter=6": "spaceBefore=11, spaceAfter=5",
    "topMargin=59,bottomMargin=58": "topMargin=57,bottomMargin=53",
}

for old, new in changes.items():
    assert s.count(old) == 1, f"Expected exactly one occurrence: {old}"
    s = s.replace(old, new)

anchor = "        cells = [[Paragraph(inline(c), styles['th' if idx == 0 else 'cell']) for c in row] for idx, row in enumerate(rowtext)]"
assert s.count(anchor) == 1, "Results table insertion point missing"

formatting = """        if len(rowtext[0]) == 2 and rowtext[0][0] == 'Metric':
            for row in rowtext[1:]:
                if row[0] in ('Accuracy', 'Precision', 'Recall'):
                    row[1] = f'{float(row[1])*100:.2f}%'
                elif row[0] in ('F1', 'ROC-AUC', 'Binary cross-entropy'):
                    row[1] = f'{float(row[1]):.4f}'
"""

s = s.replace(anchor, formatting + anchor)
p.write_text(s, encoding="utf-8")
print("Layout refinements applied")
