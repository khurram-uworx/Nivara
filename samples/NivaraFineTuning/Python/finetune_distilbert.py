"""PyTorch reference: DistilBERT fine-tuning on GLUE SST-2.

Usage:
  pip install -r requirements.txt
  python finetune_distilbert.py

This script provides reference accuracy numbers for comparison
with the Nivara C# fine-tuning implementation.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


class Sst2TorchDataset(Dataset):
    """Minimal SST-2 dataset wrapper for PyTorch."""

    def __init__(
        self, split: str, tokenizer, max_len: int = 128, data_dir: str | None = None
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        dataset = load_dataset(
            "glue", "sst2", split=split, cache_dir=data_dir
        )
        self.sentences = dataset["sentence"]
        self.labels = dataset["label"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            self.sentences[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    return (preds == labels).float().mean().item()


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch reference: DistilBERT fine-tuning on SST-2"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Cache directory for datasets (default: ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="HuggingFace model name (default: distilbert-base-uncased)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading tokenizer and model '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(
        args.model_name, num_labels=2
    )
    model = AutoModelForSequenceClassification.from_config(config)
    model.to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"Loading SST-2 dataset (max_len={args.max_len})...")
    train_dataset = Sst2TorchDataset(
        "train", tokenizer, args.max_len, args.data_dir
    )
    eval_dataset = Sst2TorchDataset(
        "validation", tokenizer, args.max_len, args.data_dir
    )
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Dev:   {len(eval_dataset)} examples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 50 == 0 or num_batches == len(train_loader):
                print(
                    f"  Batch {num_batches}/{len(train_loader)}"
                    f" - loss: {total_loss / num_batches:.4f}"
                )

        avg_train_loss = total_loss / num_batches
        print(f"  Avg training loss: {avg_train_loss:.4f}")

        model.eval()
        eval_loss = 0.0
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                eval_loss += outputs.loss.item()
                all_logits.append(outputs.logits)
                all_labels.append(labels)

        avg_eval_loss = eval_loss / len(eval_loader)
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        accuracy = compute_accuracy(logits, labels)

        print(
            f"  Dev - loss: {avg_eval_loss:.4f}"
            f", accuracy: {accuracy * 100:.2f}%"
        )

    print("\n=== Reference training complete ===")
    print(f"Best dev accuracy: {max(0.0, accuracy) * 100:.2f}%")


if __name__ == "__main__":
    main()
