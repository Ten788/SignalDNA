"""Create a traceable research evidence register from archived results."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sources = {
    "cpu_learning": "numpy_cpu_60_epochs_logged.json",
    "gpu_learning": "pytorch_cuda_60_epochs_logged.json",
    "curve_equivalence": "cpu_gpu_curve_comparison.json",
    "backend_benchmark": "cpu_gpu_benchmark_v1.json",
    "batch_scaling": "cuda_batch_scaling_v1.json",
    "precision": "cuda_precision_v1.json",
    "precision_replication": "cuda_precision_alternating_v1.json",
    "tf32": "cuda_tf32_v1.json",
    "cuda_profile": "cuda_profile_v1.json",
    "workload_scaling": "cpu_gpu_workload_scaling_v1.json",
}
evidence = {}
for key, filename in sources.items():
    path = root / "results" / filename
    raw = path.read_bytes()
    record = json.loads(raw)
    evidence[key] = {
        "path": "results/" + filename,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "schema_version": record["schema_version"],
    }

cpu = json.loads((root / "results" / sources["cpu_learning"]).read_text())
gpu = json.loads((root / "results" / sources["gpu_learning"]).read_text())
comparison = json.loads((root / "results" / sources["curve_equivalence"]).read_text())
workload = json.loads((root / "results" / sources["workload_scaling"]).read_text())
assert len(cpu["epochs"]) == len(gpu["epochs"]) == 61
assert comparison["result"] == "PASS"
assert len(workload["results"]) == 4

final = cpu["epochs"][-1]
register = {
    "schema_version": 1,
    "project": "SignalDNA",
    "purpose": "Traceable research-paper evidence",
    "sources": evidence,
    "findings": {
        "cpu_gpu_learning_curve_equivalence": {
            "source": "curve_equivalence",
            "epochs": comparison["epochs_compared"],
            "maximum_absolute_differences": comparison["maximum_absolute_differences"],
            "limitation": "Same data, initialization and training procedure; not independent generalization replication",
        },
        "final_cpu_validation": {
            "source": "cpu_learning",
            "training_accuracy": final["training_accuracy"],
            "validation_accuracy": final["validation_accuracy"],
            "validation_auc": final["validation_auc"],
            "limitation": "Validation results only; held-out test evaluation pending",
        },
        "workload_performance": {
            "source": "workload_scaling",
            "gpu_over_cpu_ratios": [
                {"examples": row["examples"], "ratio": row["gpu_over_cpu_ratio"]}
                for row in workload["results"]
            ],
            "limitation": "One machine, float64, batch 64, current implementations",
        },
    },
    "publication_status": "Evidence inventory complete; manuscript and final evaluation pending",
}
destination = root / "results" / "research_evidence_register_v1.json"
with destination.open("x", encoding="utf-8") as output:
    json.dump(register, output, indent=2)
    output.write("\n")
print(f"source_records={len(evidence)}")
print(f"tracked_findings={len(register['findings'])}")
print(f"final_validation_accuracy={final['validation_accuracy']:.6f}")
print(f"final_validation_auc={final['validation_auc']:.6f}")
print("SIGNALDNA-EVIDENCE-REGISTER-198: PASS")