# Plan — #599: joint two-host GpuProbe doc correction

**Branch:** `khurram/599` (off `khurram/552`) · **PR base:** `khurram/552` · docs-only.
**Merge chain:** `#602` (`khurram/553` → `main`) → `#608` (`khurram/552` → `khurram/553`) → this.

> As each step executes, if deferred work or a concern appears that is outside this plan,
> create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record
> its number in the GitHub issues log below — do not rely on memory or wait until the end.

## Problem

`#552` (Arc 140T) and `#553` (Iris Xe) each remeasured the eight GpuProbe legs on one host and
both **deliberately deferred the same doc corrections** — landing them from one host would
recreate the single-host-presented-as-general defect those issues exist to fix.

The probe docs still read orderings off mixed-host, cross-series ranges (e.g. `dot16` "CPU wins",
`silu` "~5–14× CPU", `gemv` "~10–34×", OpenVINO `137.9 µs`, DX12 `117–560 µs`) and treat them as
comparable. They are not: the difference is large enough to reverse conclusions, and the Arc and
Iris Xe disagree on exactly one pair.

## Evidence base (read-only; no re-measurement)

- `tests/Nivara.GpuProbe/results/552-arc-140t.md` + `.raw.txt` — Arc 140T, n=5, exit 405.
- `tests/Nivara.GpuProbe/results/553-iris-xe.md` + `.raw.txt` — Iris Xe, n=5, exit 9.
- The Iris Xe record already carries the `#607` correction (from `khurram/552`).

Measured medians `[min–max]` µs, n=5, Release, AC:

| kernel | host | CPU | DX12 | OV bf16 | OV f32 | ILGPU | Silk.NET | ComputeSharp | SYCL |
|---|---|---|---|---|---|---|---|---|---|
| `dot16` | Arc | 4.1 [2.1–18.4] | 99.5 [89.4–171.1] | 44.4 [39.6–162.3] | 39.8 [30.3–58.4] | 11.4 [8.7–17.4] | 7.0 [6.5–16.4] | 65.9 [62.7–82.3] | unavail |
| `dot16` | Xe | 7.1 [4.6–11.5] | 219.5 [140.9–249.0] | — | — | 78.8 [70.9–106.3] | 71.4 [66.8–77.4] | 146.1 [128.3–152.0] | unavail |
| `silu` | Arc | 28.1 [16.1–50.2] | 129.5 [88.6–193.9] | 28.3 [27.0–46.3] | 33.8 [21.6–38.4] | 8.6 [7.0–18.1] | 6.7 [6.2–15.3] | 68.0 [60.8–80.7] | unavail |
| `silu` | Xe | 45.7 [27.7–3233.7] | 182.7 [121.3–208.7] | — | — | 72.9 [70.8–76.5] | 69.2 [66.2–94.3] | 134.6 [122.4–164.6] | unavail |
| `gemv` | Arc | 2348.9 [817.1–2961.5] | 250.4 [178.6–342.4] | 114.0 [92.8–201.0] | 164.2 [147.9–183.0] | 144.2 [135.8–167.4] | 100.6 [90.3–103.7] | 157.5 [136.7–174.5] | unavail |
| `gemv` | Xe | 2203.2 [1127.7–11229.6] | 705.8 [633.5–751.9] | — | — | 859.7 [845.5–1212.0] | 486.2 [469.4–583.9] | 574.0 [536.3–628.2] | unavail |

**Every number written into a doc must be re-derived from the two `.raw.txt` logs in execution,
not from this table** (this table is the plan's snapshot; the logs are the source).

## Decisions taken (with the human, at plan time)

1. README table shape: **three per-kernel tables**, rows = leg, cols = `Arc 140T` / `Iris Xe`.
2. Include **`SILK.md`** (not in the issue's site list, but the README defers the `gemv` ordering
   to it and it repeats the `#607`-class claim).
3. Superseded series figures **retired from the decision table**, kept per-leg in the case studies.
4. **All four** findings beyond the issue folded into #599.
5. **No ADR**; the two-host discipline becomes a README "Method & provenance" section.
6. `docs/ACCELERATION.md:134` window is FLOP-based → restate in `TFLOP/s`.
7. `docs/ACCELERATION.md:148` → cite the Arc band, **drop** the mismatched `~5.9 T MAC/s` rate.

## The classification #599 must land (band-overlap rule; property only if both hosts agree)

Rule: two legs are *ordered* only when their bands are **disjoint**; overlap = **tie**; an ordering
is a kernel property only if it holds on both hosts.

**Holds on both hosts**
- `dot16` / `silu`: `Silk.NET < ComputeSharp`, `Silk.NET < DX12`, `ILGPU < ComputeSharp`, `ILGPU < DX12`
- `gemv`: `Silk.NET < ILGPU`, `Silk.NET < DX12`, `ComputeSharp < DX12`; `Silk.NET`, `DX12`, `ComputeSharp` all `< CPU`

**Strictly flips (device-dependent) — exactly one**
- `gemv`: `ILGPU` vs `DX12` (Arc `ILGPU 144.2 < DX12 250.4`; Xe `DX12 705.8 < ILGPU 859.7`)

**Tie on both hosts**
- `dot16`: `Silk.NET ≈ ILGPU` · `silu`: `Silk.NET ≈ ILGPU`, `CPU ≈ ILGPU`

**Ordered on one host, tie on the other (device-dependent)**
- `dot16`: `CPU` vs `Silk.NET`; `CPU` vs `ILGPU`; `ComputeSharp` vs `DX12`
- `silu`: `CPU` vs `DX12`; `CPU` vs `ComputeSharp`; `Silk.NET` vs `CPU`
- `gemv`: `CPU` vs `ILGPU`; `ILGPU` vs `ComputeSharp`; `Silk.NET` vs `ComputeSharp`

**Arc-only:** every OpenVINO pair (no runtime on the Iris Xe). **No reading at all:** SYCL.

Headline corrections: `dot16` "CPU wins" holds only on the Iris Xe (tie on the Arc); `silu`
"~5–14×" is ~4× on the Arc and **not a GPU win** on the Iris Xe (median `Silk 69.2` vs `CPU 45.7`);
`gemv` "~30×" is ~23× Arc / ~4.5× Iris Xe.

## Proposed changes — one commit per unit

1. **`tests/Nivara.GpuProbe/README.md` → two-host.**
   - Replace L77–97 (mixed-host table + † footnote) with three per-kernel tables; add a
     **Method & provenance** section (one `kernels` run per host, 5 interleaved sweeps, n=5,
     AC/Release, band-overlap = tie, property-only-if-both-hosts; source links to both artifacts).
   - Rewrite the Reading paragraph → the four-bucket ordering section above; link the exhaustive
     pairwise matrix in `results/552-arc-140t.md` as the appendix.
   - Fix the stale sites: L19 (ILGPU path verdict), L452 (OV `~26–34×`), L476 + L483–491 (ILGPU
     leg), L517 (ComputeSharp `_pending_`), L653–662 (Recommendations).
   - Exit baseline: Arc `405`, Iris Xe `9 = 3+3+3`.
2. **`docs/ILGPU.md`** — L171–179 (§5 intro + "of that run" caveat), L183–185 (margins), L193–195
   (`beat every other GPU leg`), L197–200 (`behind only OpenVINO`), L215 (matrix steady state),
   L228–232 (§7), L233–236 (tiled-gemv "OV leads"). Prose only; 3 fences untouched.
3. **`docs/ACCELERATION.md` §5** — item 4: OV bf16 gemm contrast → **Arc-only**, drop the
   mismatched rate, cite band `[92.8–201.0]` (median 114.0), link **#606**; item 12: label the
   banded-attention table to the **Iris Xe** host, link `#553`. Prose only; 4 fences untouched.
4. **`docs/ACCELERATION.md` §4** — L134 → `303–379 GMAC/s (0.61–0.76 TFLOP/s), inside the
   0.6–1.6 TFLOP/s good window`.
5. **Case-study repoint** — `OPENVINO.md` §6/matrix/Verdict (OV **Arc-only**; `~26–34×` → ~21×
   bf16 / ~14× f32; #606 at §3), `DX12.md` §5 (retire `117–560/86–364/152–525`; rewrite the
   "within noise"/"silu splits" reading per host), `COMPUTESHARP.md` §5/§6/§7 (fill the Arc medians
   65.9 / 68.0 / 157.5, clear the `pending` markers). **Plus the two human-approved citation
   repoints (folded from #610/#611):** `OPENVINO.md:287` → TensorPrimitives API page;
   `DX12.md:288` → `D3D12DDI_WAVE_MMA_DATA_0103` page.
6. **`SILK.md`** — `:46` "trails OV-bf16 on gemv" → **tie** (the `#607`-class claim), `:5` driver
   label (`1.15.37858` is the Level Zero `driverVersion`, not the Windows driver), repoint to the
   two-host tables, link #606.
7. **Series-nav link fix** — `[ILGPU](ILGPU.md)` → `[ILGPU](../../docs/ILGPU.md)` in the headers of
   `SPIRV.md`, `SILK.md`, `SYCL.md`, `DX12.md`, `OPENVINO.md`, `COMPUTESHARP.md`, plus `SILK.md:336`.

## Constraints (verified)

- `DocSnippetExtractor.MarkdownFiles()` scans only the repo root and `docs/` — **not `tests/`**, so
  the GpuProbe docs are outside the gate and free to restructure.
- **No new `.md`** under `docs/`/root → pinned `ExpectedDocumentCount = 48`, citations `106`, and
  snippets `249 / 108 gated / 141 ungated` must not move.
- `docs/ILGPU.md` (3) and `docs/ACCELERATION.md` (4) `csharp` fences preserved; no new `File.cs:NN`
  citation added in `docs/` (would move the citation pin).
- No `src/Nivara`, no kernel, no `kernels` run.

## Verification steps

1. Re-derive every README/case-study number from `552-arc-140t.raw.txt` and `553-iris-xe.raw.txt`.
2. `dotnet build tests/Nivara.Tests -c Release`.
3. Docs fixture only, on request: `dotnet test -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
   (it carries the snippet gate; ~14 s — ask first). No full suite, no perf tests.

## Blast radius

- **Files:** `tests/Nivara.GpuProbe/{README,SILK,OPENVINO,DX12,COMPUTESHARP,SPIRV,SYCL}.md`,
  `docs/{ILGPU,ACCELERATION}.md`. All documentation.
- **Code:** none. No symbol under `src/Nivara` touched; the docs reference `Kernels/CpuLeg.cs`,
  `Kernels/KernelGate.cs`, `LlamaFusedKernels.MatMulTransposedB`, `Activation.Silu` read-only.
- **Docs gates:** the pinned counts in `DocCitationTests` / `DocumentationSnippetTests` are
  unchanged by design (prose-only edits to in-scope docs; no new document).
- **Tests:** `tests/Nivara.Tests/Docs/*` is the only suite that could be affected.

## Planned commits

1. `docs(gpuprobe): carry both hosts' readings in the probe README (#599)`
2. `docs(ilgpu): scope the ILGPU readings to their host (#599)`
3. `docs(acceleration): label the GPU follow-up items to their host (#599)`
4. `docs(acceleration): correct the GEMM decision window's unit (#599)`
5. `docs(gpuprobe): repoint the case-study docs at the two-host tables (#599)`
6. `docs(gpuprobe): correct the Silk.NET gemv claim and driver label (#599)`
7. `docs(gpuprobe): fix the case-study series nav to the ILGPU doc (#599)`
8. `docs: remove TODO.md - plan executed`

## GitHub issues log

- [x] #610 — OPENVINO.md: "System.Numerics.Tensors / BFloat16" citation points at the numeric-
      guidelines page, unrelated to the label (filed during G1 grounding).
- [x] #611 — DX12.md: dead "Shader model 6.8 / Wave Matrix" citation (404 on learn.microsoft.com;
      filed during G1 grounding).
- Both folded into change unit 5 per the human (2026-10-11); the #599 PR will reference them.
