"""Generate Laya reference fixtures from the laya 0.3.20 wheel itself.

Does not transcribe the reference. Downloads the wheel, loads ``laya/common.py`` by
path with importlib (bypassing ``laya/__init__.py``, which pulls in unrelated
modules), and runs that file's ``DecisionModel``. A transcription would agree with
the C# port for the wrong reason; this one cannot, because it never sees the port.

Emits, into ``samples/data/laya/``:

  prompts_py.bin   int32, little-endian. ``n_questions``, then per question
                   ``n_ids, ids[n_ids], n_markers, markers[n_markers]``.
  logits_py.bin    float32, little-endian. Per question, the ``n_markers`` scorer
                   logits, untempered, in the same question order.
  act_py.bin       float32, little-endian. Per question, the two raw act-head
                   logits (pre-softmax), row-major ``[n_questions, 2]``.
  answers_py.json  the typed decision, both confidences, the per-option
                   probabilities, and the temperature with its raw-vs-clamped
                   provenance, so the ``choice:11+`` -> 0.5 clamp is visible.
  laya_meta.json   wheel version, state, question order, and the counts.

The C# gate (``laya compare``) reads these. Prompt ids are compared byte-exact
before any numeric comparison.
"""
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch

WHEEL = "laya==0.3.20"
STATE = "User: how do I reset my password?\nAgent: open settings, then security."

# The same four questions the C# sample scores. The gate is what keeps the two
# copies from drifting: a different instruction here fails prompt parity before
# any logit is compared.
QUESTIONS = [
    ("choice2", {
        "t": "choice",
        "ins": "Is this a login question?",
        "crit": {"reset": None, "other": "not a password question"},
    }),
    ("choice13", {
        "t": "choice",
        "ins": "Which topic?",
        "crit": {f"opt{i}": f"topic number {i}" for i in range(13)},
    }),
    ("score4", {
        "t": "score",
        "ins": "How urgent?",
        "crit": ["low", "medium", "high", "critical"],
    }),
    ("noul", {
        "t": "noul",
        "ins": "Is it resolved?",
        "labels": {"false": " no ", "true": "yes"},
        "crit": {"false": "still broken", "true": "fixed"},
    }),
]


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def download_wheel(dest: str) -> str:
    """``pip download --no-deps`` into ``dest``, returning the wheel path.

    Nothing is installed. A re-run finds the wheel already there and pip
    reports that rather than fetching it again.
    """
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading {WHEEL} (--no-deps) into {dest}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "download", WHEEL, "--no-deps", "-d", dest],
        check=True,
    )
    wheels = [f for f in os.listdir(dest) if f.startswith("laya-") and f.endswith(".whl")]
    if len(wheels) != 1:
        raise SystemExit(f"Expected exactly one laya wheel in {dest}, found {wheels}")
    return os.path.join(dest, wheels[0])


def load_common(wheel_path: str):
    """Load ``laya/common.py`` from the wheel by path, not as an installed package."""
    extract = tempfile.mkdtemp(prefix="laya_wheel_")
    with zipfile.ZipFile(wheel_path) as zf:
        zf.extractall(extract)
    common_path = os.path.join(extract, "laya", "common.py")
    if not os.path.isfile(common_path):
        raise SystemExit(f"{wheel_path} has no laya/common.py")
    spec = importlib.util.spec_from_file_location("laya_common_wheel", common_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {common_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode(common, qid: str, q: dict, logits: np.ndarray, act_logits: np.ndarray, cfg: dict) -> dict:
    """The wheel's own decode, via its functions rather than a copy of them.

    The three-way type switch is the caller's (``agent.py``), not the model's.
    Confidence, the bucket, and the clamp are ``common.py``.
    """
    k = logits.shape[0]
    qt = common.QTYPES[q["t"]]
    bucket = common.temp_bucket(qt, k)
    by_options = cfg.get("temperature_by_options", {})
    raw = by_options[bucket] if bucket in by_options else cfg["temperature"][qt]
    scale = common.clamp_temperature(raw)

    z = logits / scale
    p = np.exp(z - z.max())
    p = p / p.sum()

    answer = {
        "id": qid,
        "type": q["t"],
        "answer_confidence": round(common.answer_confidence(p, k), 4),
        "act_probability": round(float(torch.softmax(torch.tensor(act_logits), -1)[0]), 4),
        "temperature_scale": scale,
        "temperature_bucket": bucket,
        "temperature_raw": float(raw),
        "temperature_from_bucket": bucket in by_options,
        "temperature_was_clamped": scale != float(raw),
    }

    if q["t"] == "choice":
        keys = list(q["crit"].keys())
        answer["choice"] = keys[int(p.argmax())]
        answer["probabilities"] = [round(float(v), 4) for v in p]
        answer["confidence"] = round(common.confidence_from_probs(p, k), 4)
    elif q["t"] == "score":
        answer["score"] = round(float((np.arange(k) * p).sum()), 4)
        answer["probabilities"] = [round(float(v), 4) for v in p]
        answer["confidence"] = round(common.confidence_from_probs(p, k), 4)
    else:
        noul = float(p[1])
        answer["noul"] = round(noul, 4)
        answer["confidence"] = round(max(noul, 1.0 - noul), 4)
    return answer


def main() -> None:
    model_dir = os.path.join(repo_root(), "samples", "data", "laya")
    if not os.path.isfile(os.path.join(model_dir, "model.safetensors")):
        raise SystemExit(
            f"No checkpoint at {model_dir}. The fixtures are a comparison against it, "
            "so there is nothing to generate.")

    cache = os.path.join(tempfile.gettempdir(), "opencode", "laya_wheel_cache")
    try:
        wheel_path = download_wheel(cache)
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            f"Could not download {WHEEL} ({e}). The gate needs the wheel; it does not "
            "fall back to a transcription.") from e

    common = load_common(wheel_path)
    print(f"Loaded {common.__file__}")

    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    with open(os.path.join(model_dir, "rl_agent_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_dir, "tokenizer"), local_files_only=True)
    encoder_dir = os.path.join(model_dir, "encoder")
    model = common.build_model(cfg, encoder_dir=encoder_dir, pretrained=True)
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    model.eval()
    print(f"Loaded checkpoint ({len(weights)} tensors), attn={model.encoder.config._attn_implementation}")

    max_len = cfg["max_len"]
    head_max_len = cfg["head_max_len"]

    prompt_words = [len(QUESTIONS)]
    logit_rows = []
    act_rows = []
    answers = {}

    for qid, q in QUESTIONS:
        ids, markers = common.build_sequence(
            tokenizer, STATE, q, max_len=max_len, head_max_len=head_max_len)
        prompt_words.append(len(ids))
        prompt_words.extend(ids)
        prompt_words.append(len(markers))
        prompt_words.extend(markers)

        ids_t = torch.tensor([ids], dtype=torch.long)
        attn = torch.ones_like(ids_t)
        mpos = torch.tensor([markers], dtype=torch.long)
        mmask = torch.ones(1, len(markers), dtype=torch.bool)
        qtype = torch.tensor([common.QTYPES[q["t"]]], dtype=torch.long)

        with torch.no_grad():
            logits, act_logits = model(ids_t, attn, mpos, mmask, qtype)

        # B=1 and k == len(markers), so the reference's masked_fill(~marker_mask, -1e4)
        # has no padding slot to fill. Taking the row as-is is that no-op, not an omission.
        logit_row = logits[0].float().numpy()
        act_row = act_logits[0].float().numpy()
        logit_rows.append(logit_row)
        act_rows.append(act_row)

        answer = decode(common, qid, q, logit_row, act_row, cfg)
        answers[qid] = answer
        print(
            f"  {qid}: {len(ids)} tok, {len(markers)} markers, "
            f"answer={answer.get('choice', answer.get('score', answer.get('noul')))}, "
            f"temp={answer['temperature_scale']} ({answer['temperature_bucket']}"
            f"{', clamped' if answer['temperature_was_clamped'] else ''})")

    out = model_dir
    np.asarray(prompt_words, dtype=np.int32).tofile(os.path.join(out, "prompts_py.bin"))
    np.concatenate(logit_rows).astype(np.float32).tofile(os.path.join(out, "logits_py.bin"))
    np.stack(act_rows).astype(np.float32).tofile(os.path.join(out, "act_py.bin"))
    with open(os.path.join(out, "answers_py.json"), "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2)
        f.write("\n")
    meta = {
        "wheel": WHEEL,
        "state": STATE,
        "max_len": max_len,
        "head_max_len": head_max_len,
        "question_ids": [qid for qid, _ in QUESTIONS],
        "n_markers": [int(row.shape[0]) for row in logit_rows],
        "attn_implementation": model.encoder.config._attn_implementation,
    }
    with open(os.path.join(out, "laya_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"Wrote fixtures to {out}")


if __name__ == "__main__":
    main()
