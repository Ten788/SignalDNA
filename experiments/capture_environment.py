"""Capture current SignalDNA environment without accessing datasets."""
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
output = root / "results" / "runtime_environment_v1.json"
if output.exists():
    raise RuntimeError("Environment record already exists")

def command(args):
    result = subprocess.run(
        args, cwd=root, capture_output=True, text=True,
        timeout=15, check=True
    )
    return result.stdout.strip()

def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

record = {
    "schema_version": 1,
    "purpose": "Current research environment capture",
    "captured_utc": datetime.now(timezone.utc).isoformat(),
    "historical_benchmark_environment_verified": False,
    "git_head": command(["git", "rev-parse", "HEAD"]),
    "python": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "operating_system": platform.system(),
    "operating_system_release": platform.release(),
    "machine_architecture": platform.machine(),
    "numpy": np.__version__,
    "pytorch": torch.__version__,
    "pytorch_cuda_build": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "float32_matmul_precision": torch.get_float32_matmul_precision(),
    "packages": {
        name: version(name)
        for name in ("numpy", "torch", "scipy", "scikit-learn",
                     "pytest", "matplotlib", "pandas")
    },
}

if platform.system() == "Windows":
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            record["cpu_model"] = winreg.QueryValueEx(
                key, "ProcessorNameString"
            )[0].strip()
    except OSError:
        record["cpu_model"] = "unavailable"

if torch.cuda.is_available():
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append({
            "index": index,
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "compute_capability": [
                int(props.major), int(props.minor)
            ],
        })
    record["cuda_devices"] = devices
    record["cudnn_version"] = torch.backends.cudnn.version()

with output.open("x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")

print("git_head=" + record["git_head"])
print("python=" + record["python"])
print("numpy=" + record["numpy"])
print("pytorch=" + record["pytorch"])
print("cuda_build=" + str(record["pytorch_cuda_build"]))
print("cuda_available=" + str(record["cuda_available"]))
print("cpu=" + record.get("cpu_model", "unavailable"))
for device in record.get("cuda_devices", []):
    print("gpu=" + device["name"])
print("environment_record=results/runtime_environment_v1.json")
print("SIGNALDNA-ENVIRONMENT-CAPTURE-216: PASS")