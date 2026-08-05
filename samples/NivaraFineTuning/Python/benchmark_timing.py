"""Steady-state fine-tuning benchmark: PyTorch DistilBERT on SST-2.

The Nivara counterpart is `samples/NivaraFineTuning` (`--mode train`). Both
sides train DistilBERT-base (67M params) for sequence classification on the
first `--max-examples` rows of the SST-2 parquet and report steady-state
ms/batch. Run this from the sample directory:

    python Python\\benchmark_timing.py --epochs 1 --batch-size 2 --max-examples 25

Methodology (mirrors samples/NivaraInference):
  - warmup batches run untimed so JIT/oneDNN initialization settles
  - timing covers the full epoch; ms/batch = epoch time / timed batches
  - `--seed` fixes the training-batch shuffle for reproducible A/B runs
  - machine/stack info is printed so results are comparable across runs
"""
import argparse, os, platform, time, torch
import numpy as np
import pandas as pd
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Use only first N examples (0 = all rows from parquet)")
    parser.add_argument("--warmup-epochs", type=int, default=2,
                        help="Untimed epochs before timing (steady-state)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the training-batch shuffle (reproducible runs)")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"device={device}, batch_size={args.batch_size}, epochs={args.epochs}, "
          f"max_examples={args.max_examples}, warmup={args.warmup_epochs}, seed={args.seed}")
    print(f"machine={platform.platform()} | cpu={os.cpu_count()} | "
          f"python={platform.python_version()} | torch={torch.__version__} | "
          f"torch_threads={torch.get_num_threads()}")

    config = AutoConfig.from_pretrained("distilbert-base-uncased", num_labels=2)
    model = AutoModelForSequenceClassification.from_config(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"params={total_params:,}")

    data_dir = args.data_dir or os.path.join(os.path.dirname(__file__), "..", "..", "data", "sst2")
    train_path = os.path.join(data_dir, "train-00000-of-00001.parquet")
    print(f"loading {train_path}")
    df = pd.read_parquet(train_path)
    if args.max_examples > 0 and args.max_examples < len(df):
        df = df.head(args.max_examples)
    sentences = df["sentence"].tolist()
    labels = df["label"].tolist()
    print(f"train_examples={len(sentences)}")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    def run_epoch():
        model.train()
        indices = np.random.permutation(len(sentences))
        total_loss = 0.0
        num_batches = 0
        for i in range(0, len(indices), args.batch_size):
            batch_idx = indices[i:i + args.batch_size]
            batch_texts = [sentences[j] for j in batch_idx]
            batch_labels = torch.tensor([labels[j] for j in batch_idx], dtype=torch.long)
            enc = tokenizer(batch_texts, truncation=True, padding="max_length",
                            max_length=args.max_len, return_tensors="pt")
            input_ids, attn_mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return num_batches, total_loss

    # Warmup (untimed) so oneDNN/JIT initialize before measurements begin.
    for _ in range(args.warmup_epochs):
        run_epoch()

    total_batches = (len(sentences) + args.batch_size - 1) // args.batch_size
    print(f"total_batches_per_epoch={total_batches}")
    epoch_times = []
    measured_batches = 0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        num_batches, total_loss = run_epoch()
        elapsed = time.time() - epoch_start
        epoch_times.append(elapsed)
        measured_batches += num_batches
        ms_per_batch = elapsed / num_batches * 1000
        print(f"epoch {epoch}: {elapsed:.1f}s ({ms_per_batch:.1f}ms/batch, "
              f"{num_batches} batches), avg_loss={total_loss/num_batches:.4f}")

    avg_ms_per_batch = np.sum(epoch_times) / measured_batches * 1000
    full_batches = 67349 // args.batch_size  # full SST-2 train set
    extrapolated_full = avg_ms_per_batch * full_batches / 1000
    print("\n=== RESULTS ===")
    print(f"Per-epoch time ({total_batches} batches): {np.mean(epoch_times):.1f}s "
          f"({avg_ms_per_batch:.1f}ms/batch)")
    print(f"Extrapolated full epoch ({full_batches} batches): ~{extrapolated_full:.0f}s "
          f"({extrapolated_full/60:.1f}min)")
    print(f"Total measured: {sum(epoch_times):.1f}s")
    print(f"NIVARA_BENCHMARK ms_per_batch={avg_ms_per_batch:.1f} "
          f"params={total_params} seed={args.seed} warmup={args.warmup_epochs}")


if __name__ == "__main__":
    main()
