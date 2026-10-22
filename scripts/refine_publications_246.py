from pathlib import Path

changes = {
    "scripts/render_cpu_cuda_benchmark.py": [
        ('ax.text(median + 3, y,', 'ax.text(max(values) + 10, y,'),
        ('for s in series) * 1.25)', 'for s in series) * 1.40)'),
    ],
    "scripts/create_project_brief.py": [
        (
            '"A NumPy neural network built with explicit forward propagation, "\n'
            '    "backpropagation and mini-batch SGD detects Gaussian transients "\n'
            '    "in synthetic signals. The study investigates generalization, "\n'
            '    "numerical equivalence and measured CPU/GPU performance."',
            '"Built to learn neural networks from first principles and explore "\n'
            '    "future FPGA and embedded engineering applications, SignalDNA "\n'
            '    "uses a NumPy reference network to classify synthetic transients. "\n'
            '    "It examines generalization, numerical equivalence and CPU/GPU performance."'
        ),
        (
            '12,000 balanced waveforms  |  2,048 Hz  |  0.5 s  |  60 epochs',
            '12,000 balanced signals  |  1,800 held-out  |  2,048 Hz  |  0.5 s  |  60 epochs'
        ),
        (
            'David Boring  |  Independent research  |  October 2026',
            'David Boring  |  October 2026  |  Software: Apache-2.0'
        ),
    ],
}

updated = {}
for filename, replacements in changes.items():
    path = Path(filename)
    source = path.read_text(encoding="utf-8-sig")
    revised = source
    for old, new in replacements:
        if revised.count(old) == 1:
            revised = revised.replace(old, new)
        elif new not in revised:
            raise ValueError(f"Unexpected source in {filename}: {old[:55]}")
    updated[path] = (source, revised)

for path, (source, revised) in updated.items():
    if source != revised:
        backup = path.with_name(path.name + ".pre246.bak")
        if not backup.exists():
            backup.write_text(source, encoding="utf-8")
        path.write_text(revised, encoding="utf-8")
    print("VERIFIED:", path)

print("SIGNALDNA-PUBLICATION-REFINEMENT-246: PATCH PASS")
