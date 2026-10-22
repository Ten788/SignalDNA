import hashlib
import json
from pathlib import Path
import pymupdf

root = Path.cwd()
evidence = root / "results/heldout_final_primary_v1.json"
expected = "B8DC3B0EDF1054248F8245184B56FB67BB1DC5F452B770FDF710D5B34F4F7CA8"

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()

actual = sha256(evidence)
assert actual == expected, "Held-out evidence fingerprint mismatch"
print("HELD-OUT EVIDENCE: PASS")

targets = [
    ("docs/publication/SignalDNA_Research_Paper_DRAFT.pdf", 7),
    ("docs/publication/SignalDNA_Project_Brief_DRAFT.pdf", 2),
]

report = {"step": 247, "heldout_sha256": actual, "artifacts": []}

for filename, expected_pages in targets:
    path = root / filename
    assert path.is_file(), f"Missing PDF: {filename}"

    with pymupdf.open(path) as pdf:
        assert len(pdf) == expected_pages, f"Page count mismatch: {filename}"
        content = "\n".join(page.get_text() for page in pdf)
        images = sum(len(page.get_image_info()) for page in pdf)

        assert "Not peer reviewed" in content

        if expected_pages == 7:
            assert "References" in content
            assert "Figure 3." in content
            assert images >= 3
        else:
            assert "Apache-2.0" in content
            assert "1,800 held-out" in content
            assert images >= 2

        report["artifacts"].append({
            "path": filename,
            "pages": len(pdf),
            "images": images,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

    print(f"VERIFIED: {filename}")

output = root / "docs/publication/qa/publication_integrity_247.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("REPORT:", output)
print("SIGNALDNA-PUBLICATION-INTEGRITY-247: PASS")
