# SignalDNA — Optional PyTorch / CUDA Setup

Status: Pre-release installation guide. Clean-environment verification pending.

## CPU-only operation

The NumPy reference implementation and existing unittest suite do not require PyTorch or CUDA. Install the project's requirements.lock.txt in an isolated virtual environment and follow README.md.

## GPU requirements

SignalDNA's GPU experiments use PyTorch. The verified local environment snapshot recorded:

- Windows and Python 3.13.15
- NumPy 2.5.4
- PyTorch 2.11.0+cu128
- PyTorch CUDA build 12.8
- NVIDIA GeForce RTX 4070 Laptop GPU

These describe the captured research environment, not a universal installation requirement.

## Installation

1. Create and activate a dedicated project virtual environment as described in README.md.
2. Install requirements.lock.txt.
3. Open the official PyTorch installation selector:
   https://pytorch.org/get-started/locally/
4. Select Windows, Pip, Python, and a compatible CUDA compute platform.
5. Execute the command provided by PyTorch inside the project virtual environment.

Do not assume that a PyTorch CUDA wheel is compatible with every NVIDIA driver, Windows installation, or Python version. Consult the official selector for your machine.

## Verify the installed backend

From the repository root, use the project Python executable:

    .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

For an available CUDA device:

    .\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

Only run GPU experiments after these checks pass.

## Reproduction boundaries

- GPU installation is optional for CPU-only learning and unit tests.
- GPU timings depend on device characteristics, drivers, software versions, thermals, and workload.
- Installing a compatible PyTorch build does not guarantee identical benchmark timings.
- Several experimental scripts write fixed output paths. Use a disposable checkout when generating new exploratory results.
- Do not rerun the locked final held-out evaluation.
- The original benchmark evidence remains authoritative for the reported study.

## Official documentation

- https://pytorch.org/get-started/locally/
- https://docs.pytorch.org/docs/stable/notes/cuda.html