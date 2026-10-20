# Plan: GitHub issue #372 — full-model PyTorch parity for LlamaForCausalLM (tied LM head)

## Problem
`LlamaForCausalLM` (sample-side composition: embedding → N decoder blocks → final RMSNorm → tied LM head) has no PyTorch parity fixture in `samples/NivaraTorch/gen_reference.py` and no NivaraTorch parity test. Core building blocks are covered, but the full-model composition is not.

## Proposed changes
1. Append a dedicated section to `gen_reference.py` (before manifest write) using a dedicated `torch.Generator` (seed 1616) so the shared RNG streams stay bit-stable. Build the tied-head architecture with deterministic weights and save bins + manifest entry under `llama_causal_lm`.
   - Dims: vocab=128, hidden=32, layers=2, heads=4, kvHeads=2, intermediate=64, maxPos=32, eps=1e-5, ropeTheta=10000, qkvBias=false.
   - Input token ids: [1, 12, 45, 78, 99].
   - Save embedding weight, per-layer weights (in_gamma, post_gamma, qw,kw,vw,ow,gate,up,down), final-norm gamma, input ids, logits, and embedding-weight gradient.
2. Add `tests/Nivara.Tests/NivaraTorch/LlamaForCausalLMTests.cs` that loads the fixtures, constructs `LlamaForCausalLM<float>` with matching dims, builds a state dict keyed by `Module_{i}` paths, calls `LoadStateDict(strict: true)`, runs `Forward(inputIds)`, and asserts logits (and embedding gradient) parity with standard tolerance.

## Blast radius
- `samples/NivaraTorch/gen_reference.py` (additive only, appended at end; no existing fixtures perturbed because dedicated generator).
- New test file under `tests/Nivara.Tests/NivaraTorch/`.
- No changes to `src/`. No doc citation count changes.

## Verification steps
- `python gen_reference.py` runs clean; drift warning handled.
- `dotnet build Nivara.slnx`
- Targeted: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~LlamaForCausalLMTests"`

## Planned commits
1. docs: plan #372 in TODO.md
2. feat(torch): add LlamaForCausalLM tied-head reference case
3. test(torch): add LlamaForCausalLM parity test
4. docs: remove TODO.md — plan executed

## GitHub issues log
- (none yet)
