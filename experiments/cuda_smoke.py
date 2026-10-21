"""Verify CUDA execution and CPU/GPU numerical agreement."""
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))

a = torch.arange(16, dtype=torch.float32).reshape(4, 4)
expected = a @ a.T
gpu_a = a.to("cuda")
actual = gpu_a @ gpu_a.T
torch.cuda.synchronize()

assert torch.allclose(actual.cpu(), expected), "CPU/GPU result mismatch"
print("GPU result:", actual.cpu().tolist())
print("SIGNALDNA-CUDA-COMPUTE-168: PASS")