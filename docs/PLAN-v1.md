# Nivara v1 Plan

## Purpose

This document describes the path from `0.9.0` (the first public NuGet release) to a credible `v1.0.0` release.

## What 0.9.0 Achieved

- Published both `Nivara` and `Nivara.Extensions` to NuGet
- Package metadata cleaned up (SourceLink, symbols, deterministic build, repository metadata)
- Release notes and differentiated package descriptions added
- Release workflow and publish instructions documented
- Release-facing docs verified; historical idea docs kept as archive

## What v1 Should Mean

`v1.0.0` should mean the public API contract is stable enough that users can adopt the package with normal semver expectations.

For Nivara, that should imply:

- Core APIs are considered stable and intentionally versioned
- Package boundaries are clear
- Breaking changes become exceptional rather than routine
- The release process is repeatable and low-risk
- The documentation accurately reflects supported behavior

## What Still Needs To Happen Before v1

The work left before `v1.0.0` is mostly about confidence, compatibility, and maintenance discipline.

The main areas are:

- API stabilization for the core package
- Clear separation between core guarantees and extension guarantees
- Broader compatibility story if we want v1 to reach more than .NET 10 users
- Explicit extension package support stance

## v1 Readiness Criteria

Nivara is ready for `v1.0.0` when:

- The core API surface is intentionally frozen enough to support semver expectations
- The package metadata and release workflow are production-quality
- The README and package descriptions are accurate and non-promotional
- The release notes history clearly explains what changed between `0.9.0` and `v1.0.0`
- The extensions package has an explicit support stance
- The team is comfortable treating breaking changes as true major-version events

## Strategic Gap Areas

### API Contract

The biggest v1 question is not feature count. It is whether the current public API surface is ready to be promised.

The core package should settle on:

- Which types and methods are part of the stable contract
- Which behaviors are guaranteed versus implementation details
- Which extension points are permanent

### Compatibility Scope

The current target is `net10.0`.

That may be fine for `0.9.0`, but v1 should make that choice explicit and deliberate.

If broader adoption matters, the compatibility story should be decided before v1 rather than after it.

### Extension Package Stance

`Nivara.Extensions` is part of the product, but it should not define the core stability story.

For v1, the key question is:

- Is the extensions package part of the same stability promise as the core package, or a separately evolving add-on? (Current stance: separate stability.)

That decision should be explicit before the major version.

## Recommended v1 Path

1. ~~Publish `0.9.0` as the first public NuGet release~~ (done)
2. ~~Stabilize packaging, metadata, and release workflow~~ (done)
3. Tighten the public API contract and release policy
4. Decide the compatibility scope for the long-term supported line
5. Publish `1.0.0` only when the semver promise is something we are comfortable maintaining

## Non-Goals For v1

v1 should not require:

- Perfect feature completeness
- Every possible integration format
- Exhaustive optimization work
- Full cross-platform parity for every niche scenario

v1 should require:

- A stable core contract
- Clear release boundaries
- Honest documentation
- Repeatable packaging and publishing

## Summary

`0.9.0` is the public release step.

`v1.0.0` is the stability promise step.

The work between them is mostly not feature work. It is about making the current system into a package people can trust, consume, and maintain against a clear contract.
