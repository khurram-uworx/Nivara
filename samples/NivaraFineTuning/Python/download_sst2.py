"""Download GLUE SST-2 dataset from HuggingFace mirror and extract to samples/data/sst2/."""

import argparse
import os
import sys
import zipfile
import urllib.request

SST2_URL = "https://huggingface.co/datasets/nyu-mll/glue/resolve/main/data/SST-2.zip"


def download_sst2(data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "SST-2.zip")

    if os.path.exists(zip_path):
        print(f"Found existing download at {zip_path}, skipping download.")
    else:
        print(f"Downloading SST-2 from {SST2_URL}...")
        urllib.request.urlretrieve(SST2_URL, zip_path)
        print("Download complete.")

    extract_dir = os.path.join(data_dir)
    print(f"Extracting to {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    train_tsv = os.path.join(extract_dir, "SST-2", "train.tsv")
    dev_tsv = os.path.join(extract_dir, "SST-2", "dev.tsv")

    if not os.path.exists(train_tsv):
        print(f"ERROR: Expected {train_tsv} not found after extraction.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(dev_tsv):
        print(f"ERROR: Expected {dev_tsv} not found after extraction.", file=sys.stderr)
        sys.exit(1)

    # Optionally move files up a level for convenience
    final_train = os.path.join(extract_dir, "train.tsv")
    final_dev = os.path.join(extract_dir, "dev.tsv")
    if not os.path.exists(final_train):
        os.rename(train_tsv, final_train)
    if not os.path.exists(final_dev):
        os.rename(dev_tsv, final_dev)

    # Clean up zip
    os.remove(zip_path)
    # Remove SST-2 subdirectory if empty
    sst2_subdir = os.path.join(extract_dir, "SST-2")
    if os.path.isdir(sst2_subdir):
        try:
            os.rmdir(sst2_subdir)
        except OSError:
            pass  # not empty, leave it

    print(f"SST-2 data ready at {extract_dir}")
    print(f"  train.tsv: {os.path.getsize(final_train):,} bytes")
    print(f"  dev.tsv:   {os.path.getsize(final_dev):,} bytes")


def main():
    parser = argparse.ArgumentParser(description="Download GLUE SST-2 dataset")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "sst2"),
        help="Output directory for SST-2 data (default: samples/data/sst2/)",
    )
    args = parser.parse_args()
    download_sst2(os.path.abspath(args.data_dir))


if __name__ == "__main__":
    main()
