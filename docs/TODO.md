# Plan: Issue #455 — gen_reference.py byte-reproducibility / provenance

## Problem
`gen_reference.py` is not byte-reproducible against the committed fixture tree: 35 of 308 `.bin` files drift on re-run due to torch build/MKL/microarch differences in reductions. Two consecutive runs on the same machine are identical, so it is not flakiness. There is no machine-readable record of which environment produced the committed tree, so the churn is unattributable.

## Proposed changes
1. `samples/NivaraTorch/gen_reference.py`
   - Add `collect_provenance()` capturing python, platform, processor, torch + git version, numpy, thread counts, env thread vars, CPU capability, sha256 + full text of `torch.__config__.show()`.
   - Stamp `manifest["provenance"]` and `manifest["environment"]` (alias).
   - Print one-line provenance summary at startup.
   - Add drift warning comparing committed manifest provenance to current.
   - Correct docstring reproducibility wording.
2. `samples/NivaraTorch/requirements.txt` — pin torch/numpy.
3. `samples/NivaraTorch/README.md` — correct wording, document provenance, update versions.

## Verification steps
- Run generator twice; manifest.json byte-identical.
- Drift warning prints on first run against old manifest; none on second.
- `git status` after restore shows only manifest.json.

## Planned commits
1. docs: plan #455 in TODO.md
2. feat: add machine-readable provenance stamp to gen_reference.py
3. docs: pin torch environment and document provenance in README

## GitHub issues log
- (none yet)
