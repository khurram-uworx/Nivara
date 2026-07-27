"""
Forward-Mode JVP Parity — PyTorch reference for Nivara.

Computes Jacobian-Vector Products (JVPs) for a set of test functions
using PyTorch's forward-mode AD (torch.autograd.forward_ad), then
saves inputs, seeds, and JVP results to JSON for Nivara to validate.

Usage:
    python samples/pytorch/forward_parity_pytorch.py

Output:
    samples/data/jvp_cases.json  — consumed by ForwardParityExample (Nivara)
"""

import json
import os
import torch
from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def compute_jvp(fn, inputs, tangents):
    """
    Compute JVP of fn at inputs in direction tangents.
    Returns (primal, jvp) as flat float lists.
    """
    with dual_level():
        duals = []
        for x, v in zip(inputs, tangents):
            x_t = torch.tensor(x, dtype=torch.float64, requires_grad=False)
            if v is None:
                v = [0.0] * len(x)
            duals.append(make_dual(x_t, torch.tensor(v, dtype=torch.float64)))

        result = fn(*duals)
        dual_out = unpack_dual(result)
        primal = dual_out.primal.detach().flatten().tolist()
        tangent = dual_out.tangent.flatten().tolist()
        return primal, tangent


# ── Test cases ──────────────────────────────────────────────────────────────

test_cases = []

# --- Test 1: Square (element-wise multiply) ---
def square(x):
    return x * x

p, j = compute_jvp(square, [[2.0, 3.0]], [[1.0, 0.0]])
test_cases.append({
    "name": "square",
    "description": "f(x) = x*x, seed=[1,0] (derivative w.r.t. first element only)",
    "inputs": {"x": [2.0, 3.0]},
    "seeds": {"x": [1.0, 0.0]},
    "primal": p,
    "jvp": j,
})

# --- Test 2: Multiply + Add (binary ops chain) ---
def mul_add(a, b):
    return a * b + a

p, j = compute_jvp(mul_add, [[1.5], [2.5]], [[1.0], [1.0]])
test_cases.append({
    "name": "mul_add",
    "description": "f(a,b) = a*b + a, seed on a=[1], b=[1]",
    "inputs": {"a": [1.5], "b": [2.5]},
    "seeds": {"a": [1.0], "b": [1.0]},
    "primal": p,
    "jvp": j,
})

# --- Test 3: ReLU (activation with boundary at 0) ---
def relu_fn(x):
    return torch.nn.functional.relu(x)

p, j = compute_jvp(relu_fn, [[-1.0, 0.0, 2.0]], [[1.0, 1.0, 1.0]])
test_cases.append({
    "name": "relu",
    "description": "f(x) = relu(x), seed=[1,1,1]",
    "inputs": {"x": [-1.0, 0.0, 2.0]},
    "seeds": {"x": [1.0, 1.0, 1.0]},
    "primal": p,
    "jvp": j,
})

# --- Test 4: Sigmoid (activation) ---
def sigmoid_fn(x):
    return torch.nn.functional.sigmoid(x)

p, j = compute_jvp(sigmoid_fn, [[0.0, 1.0]], [[1.0, 1.0]])
test_cases.append({
    "name": "sigmoid",
    "description": "f(x) = sigmoid(x), seed=[1,1]",
    "inputs": {"x": [0.0, 1.0]},
    "seeds": {"x": [1.0, 1.0]},
    "primal": p,
    "jvp": j,
})

# --- Test 5: MatMul (matrix-vector product with fixed W) ---
# W is constant inside the function, only x is a free input
W_mm = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
def matmul_fn(x):
    return W_mm @ x

p, j = compute_jvp(matmul_fn, [[1.0, 2.0]], [[1.0, 0.0]])
test_cases.append({
    "name": "matmul",
    "description": "f(x) = W@x, W fixed=[[1,2],[3,4]], seed=[1,0] (JVP = W[:,0] = [1,3])",
    "inputs": {"x": [1.0, 2.0]},
    "seeds": {"x": [1.0, 0.0]},
    "primal": p,
    "jvp": j,
})

# --- Test 6: Full composition: f(x) = sum(relu(W@x + b)) ---
def composition(x):
    W = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float64)
    b = torch.tensor([0.1, -0.1], dtype=torch.float64)
    h = torch.nn.functional.relu(W @ x + b)
    return h.sum()

p, j = compute_jvp(composition, [[1.0, -0.5]], [[1.0, 0.0]])
test_cases.append({
    "name": "composition",
    "description": "f(x) = sum(relu(W@x + b)), seed=[1,0], W=[[0.5,-0.5],[-0.5,0.5]], b=[0.1,-0.1]",
    "inputs": {"x": [1.0, -0.5]},
    "seeds": {"x": [1.0, 0.0]},
    "primal": p,
    "jvp": j,
})


# ── Save to JSON ────────────────────────────────────────────────────────────

output = {
    "test_cases": test_cases,
}

out_path = os.path.join(OUT_DIR, "jvp_cases.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved {len(test_cases)} test cases to {out_path}")
for tc in test_cases:
    print(f"  {tc['name']}: primal={tc['primal']}, jvp={tc['jvp']}")
print("\nDone. Run the Nivara ForwardParityExample to validate.")
