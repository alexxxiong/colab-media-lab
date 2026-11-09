"""Small, synthetic LoRA training check; uses Colab's existing PyTorch only."""
import json
import time
import torch

started = time.monotonic()
assert torch.cuda.is_available(), "CUDA is unavailable"
device_name = torch.cuda.get_device_name(0)
assert "T4" in device_name.upper(), f"Requested T4, received {device_name}; stopping"
torch.manual_seed(42)
device = torch.device("cuda")
width, rank = 64, 4
base = torch.nn.Linear(width, width, bias=False, device=device)
base.requires_grad_(False)
base_before = base.weight.detach().clone()
adapter_a = torch.nn.Linear(width, rank, bias=False, device=device)
adapter_b = torch.nn.Linear(rank, width, bias=False, device=device)
torch.nn.init.zeros_(adapter_b.weight)
x = torch.randn(128, width, device=device)
with torch.no_grad():
    residual = (x @ torch.randn(width, rank, device=device)) @ torch.randn(rank, width, device=device) / 16
    target = base(x) + residual

def loss():
    return torch.nn.functional.mse_loss(base(x) + adapter_b(adapter_a(x)), target)

before = loss().item()
optimizer = torch.optim.AdamW([*adapter_a.parameters(), *adapter_b.parameters()], lr=0.03)
for _ in range(40):
    optimizer.zero_grad(set_to_none=True)
    value = loss()
    assert torch.isfinite(value).item()
    value.backward()
    optimizer.step()
after = loss().item()
assert after < before * 0.7, f"Loss did not improve enough: {before} -> {after}"
assert torch.equal(base.weight, base_before), "Frozen base was modified"
assert base.weight.grad is None
torch.cuda.synchronize()
print("SMOKE_RESULT=" + json.dumps({
    "passed": True,
    "gpu": device_name,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "trainable_parameters": sum(p.numel() for m in (adapter_a, adapter_b) for p in m.parameters()),
    "initial_loss": before,
    "final_loss": after,
    "elapsed_seconds": round(time.monotonic() - started, 3),
    "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
}))
