#!/usr/bin/env python3
"""Convert float weights (F32/F16) in a SafeTensors file to BF16.

Nivara's `SafeTensorsLoader` is dtype-aware: a BF16 file read via the default
`Read()` is widened losslessly to F32, and `Read<BFloat16>()` reinterprets the
raw bytes directly (the zero-hop `ConvertBF16ToBFloat16` path). The sample's
models all ship in F32, so to exercise the BF16-on-disk + zero-hop path you must
first convert a checkpoint to BF16. This script rewrites F32/F16 tensors as BF16
(truncating each float32 to its top 16 bits, the canonical BF16 representation),
leaving I32/I64 tensors untouched and preserving tensor names, shapes, and the
offset layout (so the C# loaders keep mapping keys 1:1).

Usage:
    python convert_f32_to_bf16.py <model.safetensors> [--out out.safetensors]
    python convert_f32_to_bf16.py <model.safetensors> --in-place

Then run the sample's bf16 mode against the same path:
    dotnet run --project samples/NivaraInference -c Release -- distilbert_sst bf16
"""
import json
import struct
import sys

import numpy as np

FLOAT_DTYPES = {"F32", "F16", "BF16"}
DTYPE_SIZE = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4, "I64": 8}


def f32_bytes_to_bf16(raw: bytes) -> bytes:
    u32 = np.frombuffer(raw, dtype="<f4").view("<u4")
    u16 = (u32 >> 16).astype("<u2")
    return u16.tobytes()


def f16_bytes_to_bf16(raw: bytes) -> bytes:
    arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    return f32_bytes_to_bf16(arr.tobytes())


def convert(in_path: str, out_path: str) -> None:
    with open(in_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data = f.read()

    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    converted = {}
    for name, meta in tensors.items():
        dtype = meta["dtype"]
        if dtype not in DTYPE_SIZE:
            raise ValueError(f"Unsupported dtype {dtype} for tensor {name}")
        begin, end = meta["data_offsets"]
        raw = data[begin:end]
        if dtype == "F32":
            converted[name] = ("BF16", f32_bytes_to_bf16(raw))
        elif dtype == "F16":
            converted[name] = ("BF16", f16_bytes_to_bf16(raw))
        else:
            converted[name] = (dtype, raw)

    offset = 0
    new_header = {}
    for name, (dtype, raw) in converted.items():
        new_header[name] = {
            "dtype": dtype,
            "shape": tensors[name]["shape"],
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
    if "__metadata__" in header:
        new_header["__metadata__"] = header["__metadata__"]

    payload = json.dumps(new_header).encode("utf-8")
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(payload)))
        f.write(payload)
        for _, (_, raw) in converted.items():
            f.write(raw)

    n_bf16 = sum(1 for _, (d, _) in converted.items() if d == "BF16")
    print(f"Wrote {out_path}: {len(converted)} tensors ({n_bf16} -> BF16), "
          f"{offset} bytes of tensor data")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = None
    if "--in-place" in sys.argv:
        out_path = in_path
    elif "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]
    if out_path is None:
        out_path = in_path + ".bf16"
    convert(in_path, out_path)
