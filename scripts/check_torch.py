"""Quick sanity check for the Python / PyTorch / CUDA stack."""

import sys
import torch

print(f"Python:         {sys.version.split()[0]}")
print(f"PyTorch:        {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device:    {torch.cuda.get_device_name(0)}")
else:
    print("CUDA device:    CPU only (laptop will train on CPU)")
mps = getattr(torch.backends, "mps", None)
if mps is not None:
    print(f"MPS available:  {mps.is_available()}")
