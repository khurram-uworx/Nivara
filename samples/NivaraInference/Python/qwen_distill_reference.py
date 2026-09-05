"""Ground-truth fixture for the Qwen teacher-distillation student parity test.

The NivaraInference `qwen distill` mode trains a tiny student classifier over
teacher-annotated sentiment data. Its forward path is the composed MLP

    Linear(4096 -> 64) -> ReLU -> Linear(64 -> 2) -> CrossEntropyLoss(mean)

built from the same public building blocks Nivara pins elsewhere (linear_128_64_*,
relu_1d_*, cross_entropy_* fixtures). This script pins the COMPOSITION end-to-end
against PyTorch: with fixed seeded weights it writes a small, model-independent
fixture set (no Qwen checkpoint needed) so the C# test can compare forward logits,
the mean cross-entropy loss, and the first-layer weight gradient after one backward.

Run (any Torch env, from the repo root):

    python samples/NivaraInference/Python/qwen_distill_reference.py

Writes float32 dump files (little-endian) to samples/data/qwen-distill/:

    l1_w.bin, l1_b.bin      Linear(4096->64) weight [64, 4096] row-major + bias [64]
    l2_w.bin, l2_b.bin      Linear(64->2)     weight [2, 64] row-major   + bias [2]
    x.bin                   input features [B, 4096] row-major
    t.bin                   int32 class targets [B] (0/1)
    logits.bin              forward logits [B, 2] row-major   (l2(relu(l1(x))))
    loss.bin                scalar mean cross-entropy loss (one float32)
    grad_l1_w.bin           dLoss/dW1 after one loss.backward(), shape [64, 4096]

The fixtures are intentionally committed so the NUnit parity test
(tests/Nivara.Tests/Qwen/QwenDistillStudentParityTests.cs) runs in CI without the
989 MB Qwen checkpoint; the test silently Assert.Ignore()s when they are absent.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH = 4
FEAT_DIM = 4096
HIDDEN = 64
CLASSES = 2
SEED = 2026


def dump(name: str, arr: np.ndarray) -> None:
    path = os.path.join(OUT_DIR, name)
    if np.issubdtype(arr.dtype, np.integer):
        # torch.long targets are int64; the fixture contract pins int32.
        arr = arr.astype(np.int32, copy=False)
        arr.tofile(path)
    else:
        arr.astype(np.float32, copy=False).tofile(path)
    print(f"  wrote {name}  {arr.shape}  {os.path.getsize(path)} B")


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "qwen-distill")
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

l1 = nn.Linear(FEAT_DIM, HIDDEN)
l2 = nn.Linear(HIDDEN, CLASSES)

x = torch.randn(BATCH, FEAT_DIM)
targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)

logits = l2(F.relu(l1(x)))
loss = F.cross_entropy(logits, targets)  # default reduction='mean' (batch average)
loss.backward()

assert l1.weight.grad is not None, "backward must populate l1.weight.grad"

print(f"seed={SEED}, batch={BATCH}, features={FEAT_DIM}, hidden={HIDDEN}, classes={CLASSES}")
print(f"mean CE loss = {loss.item():.9f}")
print(f"forward logits max |.| = {logits.detach().abs().max().item():.6f}")
print(f"grad_l1_w max |.| = {l1.weight.grad.abs().max().item():.6f}")
print(f"writing fixtures to {OUT_DIR}")

dump("l1_w.bin", l1.weight.detach().numpy())        # [64, 4096]
dump("l1_b.bin", l1.bias.detach().numpy())          # [64]
dump("l2_w.bin", l2.weight.detach().numpy())        # [2, 64]
dump("l2_b.bin", l2.bias.detach().numpy())          # [2]
dump("x.bin", x.numpy())                            # [4, 4096]
dump("t.bin", targets.numpy())                      # [4] int32
dump("logits.bin", logits.detach().numpy())         # [4, 2]
dump("loss.bin", np.array([loss.item()]))           # scalar
dump("grad_l1_w.bin", l1.weight.grad.numpy())       # [64, 4096]

print("done.")