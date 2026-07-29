"""Quick timing benchmark: DistilBERT fine-tuning on SST-2.
Run with --max-examples N for a quick estimate instead of full epoch.
"""
import argparse, os, time, torch, numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Use only first N examples (0 = all rows from parquet)")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"device={device}, batch_size={args.batch_size}, epochs={args.epochs}")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    config = AutoConfig.from_pretrained("distilbert-base-uncased", num_labels=2)
    model = AutoModelForSequenceClassification.from_config(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"params={total_params:,}")

    # Load SST-2 from local parquet, optionally limit examples
    data_dir = args.data_dir or os.path.join(os.path.dirname(__file__), "..", "..", "data", "sst2")
    train_path = os.path.join(data_dir, "train-00000-of-00001.parquet")
    print(f"loading {train_path}")
    df = pd.read_parquet(train_path)
    if args.max_examples > 0 and args.max_examples < len(df):
        df = df.head(args.max_examples)
    sentences = df["sentence"].tolist()
    labels = df["label"].tolist()
    print(f"train_examples={len(sentences)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    total_batches = len(sentences) // args.batch_size
    print(f"total_batches_per_epoch={total_batches}")

    epoch_times = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
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

            report_every = max(1, total_batches // 5)
            if num_batches % report_every == 0:
                print(f"  batch {num_batches}/{total_batches}, loss={total_loss/num_batches:.4f}")

        elapsed = time.time() - epoch_start
        epoch_times.append(elapsed)
        ms_per_batch = elapsed / num_batches * 1000
        print(f"epoch {epoch}: {elapsed:.1f}s ({ms_per_batch:.1f}ms/batch, {num_batches} batches), avg_loss={total_loss/num_batches:.4f}")

    avg_epoch = np.mean(epoch_times)
    avg_ms_per_batch = avg_epoch / total_batches * 1000
    full_batches = 67349 // args.batch_size  # full SST-2 train set
    extrapolated_full = avg_ms_per_batch * full_batches / 1000
    print(f"\n=== RESULTS ===")
    print(f"Per-epoch time ({total_batches} batches): {avg_epoch:.1f}s ({avg_ms_per_batch:.1f}ms/batch)")
    print(f"Extrapolated full epoch ({full_batches} batches): ~{extrapolated_full:.0f}s ({extrapolated_full/60:.1f}min)")
    print(f"Total measured: {sum(epoch_times):.1f}s")

if __name__ == "__main__":
    main()
