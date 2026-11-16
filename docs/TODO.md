# Plan: Issue #1227 — Laya generic ParameterLine<TWeight>

## Problem

`docs/plan/LAYA-HONEST-WEIGHTS.md` §6 asked for a generic `ParameterLine<TWeight>` deriving label and
bytes-per-element from `typeof(TWeight)` / `Unsafe.SizeOf<TWeight>()`, called from all CPU/GPU paths,
so the native run prints:

`Parameters: 421,293,830 (803.6 MB as Half / 1607.1 MB as F32)`

Today the CPU paths (`RunInference`, `RunBenchmark` in `samples/NivaraInference/Laya.cs`) print no
parameter line, and `RunGpu` (BF16/Half in `Laya.Narrow.cs`) plus the F32 GPU path print only the
single-dtype form.

## Proposed changes

1. Replace the three `ParameterLine` overloads (`float[]`, `BFloat16[]`, `Half[]` dictionaries) with
   one generic `ParameterLine<TWeight>(Dictionary<string,(TWeight[] Data,int[] Shape)> tensors,
   bool includeF32Equivalent = false)` computing `parameters * Unsafe.SizeOf<TWeight>()` and labelling
   the storage dtype from `typeof(TWeight)` (float→"F32", Half→"Half", BFloat16→"BFloat16").
2. When `includeF32Equivalent` is true (native case), append ` / {parameters*4} MB as F32`.
3. Call it from `RunInference`, `RunBenchmark` (CPU), both narrow `RunGpu` overloads, and the F32 GPU
   path (already called there).

## Verification

- Build solution.
- Run `dotnet NivaraInference.dll laya` and confirm `Parameters: 421,293,830 (803.6 MB as Half / 1607.1 MB as F32)`.
- `laya --precision f32` → `(1607.1 MB as F32)`.
- `laya --precision fp16` → `(803.6 MB as Half)`.
- Targeted `InferenceCliArgumentTests`.

## Planned commits

1. docs: plan #1227 in TODO.md
2. refactor: generic ParameterLine<TWeight> + call sites

## Blast radius

- `samples/NivaraInference/Laya.cs`, `samples/NivaraInference/Laya.Narrow.cs`.
- No core library changes; no test edits expected.

## GitHub issues log

- (none yet)
