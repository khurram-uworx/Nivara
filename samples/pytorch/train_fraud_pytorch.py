"""
PyTorch FraudNet — cross-framework parity example.

Trains a 3-layer MLP on synthetic fraud data with same architecture as
Nivara's Act 8 FraudNet. Saves initial weights, trained weights, and
test predictions for cross-framework comparison.
"""
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_COLS = [
    "amount", "hour", "distance", "prev_attempts",
    "country_change", "device_new", "amount_ratio", "velocity",
]


class FraudNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(8, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 1)

    def forward(self, x):
        h = torch.relu(self.l1(x))
        h = torch.relu(self.l2(h))
        return self.l3(h)


def param_name(prefix, p_name):
    layer_map = {"l1": "L1", "l2": "L2", "l3": "L3"}
    p_map = {"weight": "Weight", "bias": "Bias"}
    return f"{layer_map[prefix]}.{p_map[p_name]}"


def load_csv(path, means=None, stds=None):
    df = pd.read_csv(path)
    X = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
    if means is not None and stds is not None:
        X = (X - means) / stds
    y = torch.tensor(df["is_fraud"].values, dtype=torch.float32).view(-1, 1)
    return X, y, df


def save_json_weights(model, path):
    state = {}
    for prefix, module in model.named_children():
        for p_name, param in module.named_parameters():
            key = param_name(prefix, p_name)
            state[key] = param.detach().cpu().numpy().tolist()
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_json_weights(model, path):
    with open(path, "r") as f:
        state = json.load(f)
    for prefix, module in model.named_children():
        for p_name, param in module.named_parameters():
            key = param_name(prefix, p_name)
            if key in state:
                arr = np.array(state[key], dtype=np.float32)
                param.data.copy_(torch.tensor(arr).reshape(param.shape))


# ── Load data ──────────────────────────────────────────────────────
X_train, y_train, train_df = load_csv(os.path.join(DATA_DIR, "train_fraud.csv"))
X_test, y_test, test_df = load_csv(os.path.join(DATA_DIR, "test_fraud.csv"))

means = X_train.mean(dim=0, keepdim=True)
stds = X_train.std(dim=0, keepdim=True)
stds = torch.where(stds < 1e-8, torch.ones_like(stds), stds)

norm_params = {"means": means.tolist()[0], "stds": stds.tolist()[0]}
norm_path = os.path.join(DATA_DIR, "norm_params.json")
with open(norm_path, "w") as f:
    json.dump(norm_params, f, indent=2)
print(f"Saved normalization params to {norm_path}")

X_train = (X_train - means) / stds
X_test = (X_test - means) / stds

# ── Init model and save initial weights ────────────────────────────
model = FraudNet()
init_weights_path = os.path.join(DATA_DIR, "initial_weights.json")
save_json_weights(model, init_weights_path)
print(f"Saved initial weights to {init_weights_path}")

# ── Training ───────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
batch_size = 32
epochs = 50
n = len(X_train)

epoch_losses = []
for epoch in range(1, epochs + 1):
    perm = torch.arange(n)
    epoch_loss = 0.0
    num_batches = 0
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        Xb, yb = X_train[idx], y_train[idx]

        logits = model(Xb)
        loss = loss_fn(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg = epoch_loss / num_batches
    epoch_losses.append(avg)
    print(f"Epoch {epoch:3d} | Loss: {avg:.6f} | Batches: {num_batches}")

# ── Save trained weights ──────────────────────────────────────────
trained_path = os.path.join(OUT_DIR, "trained_weights_pytorch.json")
save_json_weights(model, trained_path)
print(f"Saved trained weights to {trained_path}")

# ── Save epoch losses ─────────────────────────────────────────────
losses_path = os.path.join(OUT_DIR, "epoch_losses_pytorch.json")
with open(losses_path, "w") as f:
    json.dump(epoch_losses, f, indent=2)

# ── Inference on test set ─────────────────────────────────────────
model.eval()
with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits).cpu().numpy().flatten()

test_preds_path = os.path.join(OUT_DIR, "test_preds_pytorch.csv")
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_fraud.csv"))
test_df["logit"] = logits.cpu().numpy().flatten()
test_df["prob"] = probs
test_df.to_csv(test_preds_path, index=False)
print(f"Saved test predictions to {test_preds_path}")
print(f"Final test logits (first 5): {logits[:5].cpu().numpy().flatten().round(4)}")
print(f"Final test probs  (first 5): {probs[:5].round(4)}")
