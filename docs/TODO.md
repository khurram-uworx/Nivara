# Plan: Silk conv classifier — device forward, host-gradient training, then measure (#901)

Branch: `khurram/900` (execution vehicle for #901's groundwork; #900 owns device-side backward kernels).

## Problem

`GpuTextConvClassifier` currently (a) runs the heaviest work (embedding + three conv branches)
on the host in scalar loops while the device only courts FC1/FC2 GEMMs, and (b) "trains" by
running a CPU companion autograd model and copying weights to the device afterwards. The #901
benchmark cannot answer "is Silk worth it for the conv classifier" until the GPU path is a
complete, honest forward and training uses the same path it optimizes.

## Changes (agreed scope with human: full-device forward + host-gradient training)

1. **Device-side conv forward** in `GpuTextConvClassifier.Forward`:
   - Keep embedding gather on host; upload `[B, E, S]` batch input.
   - Upload conv branch weights/biases once in ctor/`SetWeights`.
   - One shared `convOut` buffer `[B, 3·64·S]`; per-branch `Conv2d` kernels
     (`N=B, C=E, H=1, W=S, outC=64, kH=1, kW=k, stride=1, pad=(k-1)/2, groups=1`) write into
     element-offset subviews — NCHW branch output = the flat concat layout `GemmRelu` expects,
     so the concat copy disappears.
   - One `Relu` launch over `convOut`, then `GemmRelu` → `GemmBias` → download logits.
   - Persistent scratch buffers (Ensure-grown), no per-call Allocate/Release churn.
   - Delete host `ConvBranch` and per-sample allocations.
2. **Host-gradient conv training** replacing `HostConvModel`/`TrainingLoop`:
   - Hand-written batched backward: FC2 grads + dH (ReLU mask from post-activation values),
     FC1 grads (gradW1 `[convHidden×hidden]`) + dConvOut, per-branch conv weight grads +
     dInput accumulation, embedding scatter with mean-pool scale.
   - Host Adam (first/second moments) preserving the sample's optimizer recipe (decision
     raised with human; SGD symmetry rejected in favor of keeping quality parity with the
     CPU Adam baseline).
   - `Train` returns per-epoch summary (elapsed + loss).
   - Delete `HostConvModel`.
3. **Layout overloads**: `SetWeightsCpuLayout`/`Export*Cpu` so `Program.cs` drops its
   hand-rolled GPU↔CPU index transposes (train and predict paths).
4. **Timing seam**: public `SilkForwardTimings` (host/upload/device/download ms + calls) with
   `ResetTimings()` on both classifiers, accumulated via Stopwatch inside Forward.
5. **Linear-class hot-path cleanup**: remove dead `embBuf` (uploaded per batch, never bound to
   a kernel), persistent forward scratch + `gradEmb` buffers.
6. **Bench command**: `--command bench` (+ `--bench-iterations`, `--bench-batch-sizes`) in
   `NivaraClassifier` — warm-up + interleaved CPU/Silk rounds, median ratio, per-stage transfer
   breakdown, device string, printed enable-heuristic verdict.
7. **Docs**: `samples/NivaraClassifier/README.md` measurements + heuristic;
   `docs/SILK.md` conv classifier section (device forward, host backward state);
   `docs/AUTODIFF.md` where the classifier training narrative is affected.

## Tests (added in the same commit as each change)

- `tests/Nivara.Tests/Gpu/GpuTextConvClassifierTests.cs`:
  - Device forward parity vs autograd reference (`Embedding/Conv1d/Linear` modules) on identical
    weights — tolerance `1e-5f`, rationale documented in fixture remarks (cross-kernel
    reassociation; not a bit-reproduction claim). Gated by the `Device()` → `Assert.Ignore`
    pattern used by the other Silk fixtures.
  - Gradient parity (GPU-free): hand-written backward vs `ReverseGradTensor` autograd backward,
    bounded assertion with rationale.
  - `SetWeights`/`Export*` round-trip slicing checks per #900's list.
- CPU-vs-Silk accuracy smoke via the sample CLI (small `--num-samples --epochs`).

## Verification

- `dotnet build Nivara.slnx` per step; targeted `dotnet test -c Release --filter
  "FullyQualifiedName~Nivara.Tests.Gpu"` per step (after build).
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` before push.
- Docs namespace gates (`Nivara.Tests.Docs`) since SILK.md/AUTODIFF.md change — revise
  `DocsCoveragePins` only if the arithmetic genuinely shifts; reduced coverage is a failure.
- Small smoke runs of the CLI; full benchmark runs left to the human.

## Blast radius

- `src/Nivara.Extensions/Silk/GpuTextConvClassifier.cs` — API surface changes (Train return
  type, new overloads, new exported Export* names). Public sample-serving type; only known
  caller is `samples/NivaraClassifier/Program.cs`.
- `src/Nivara.Extensions/Silk/GpuTextLinearClassifier.cs` — dead `embBuf` removal; signature
  unchanged; only caller Program.cs.
- `samples/NivaraClassifier/Program.cs` + README.
- `src/Nivara.Extensions/Silk/SilkForwardTimings.cs` (new).
- `docs/SILK.md`, `docs/AUTODIFF.md` → `DocsCoveragePins` arithmetic + snippet gate in scope.
- Tests: new Gpu fixture; no existing tests touch these classes (verified by symbol search).
- CI: `--filter Category!=Performance` unaffected; no timing tests added so far.

## Planned commits

1. `docs: plan Silk conv classifier work in TODO.md`
2. Device conv forward + parity test (+ ConvBranch removal)
3. Host-gradient conv training + Adam + gradient-parity test + Program.cs layout overloads
4. SilkForwardTimings on both classifiers
5. Linear-class transfer cleanup
6. Bench command + README measurements
7. SILK.md/AUTODIFF.md updates (+ DocsCoveragePins revision if needed)
8. `docs: remove TODO.md — plan executed`

## GitHub issues log

- [ ] #900 — device-side backward kernels and parity coverage (pre-existing; this branch lands
      forward completeness and host-gradient training as its prerequisite; conv1d forward and
      FC parity tests here reduce its remaining scope)
- [ ] #901 — measure/verify Silk classifier performance (this branch's parent goal; bench
      command lands here but canonical measured numbers expect the human's full runs)
