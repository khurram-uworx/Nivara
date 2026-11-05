# Plan: RagIndexCache content hash + schema version (#922)

## Problem
`RagIndexCache` invalidation relies on source-doc max mtime (+ chunk count in the
manifest). A doc edited and then restored to its original mtime leaves a stale
cache. There is also no schema version, so a future cache-format change could
silently misread an old file.

## Proposed changes
- Add `ContentHash` (SHA256 over concatenated source file bytes) to `Manifest`.
- Add `SchemaVersion` (int) to `Manifest`; bump on format change; mismatched
  version -> rebuild.
- Compare both on load.
- Keep existing backend/docsDir/max-mtime checks.

Files:
- `samples/NivaraChat/Helpers/RagIndexCache.cs`
- tests: `tests/Nivara.Tests/Samples/RagIndexCacheTests.cs` (extend if needed)

## Blast radius
- `RagIndexCache.Manifest` record gains two fields (with defaults for
  deserialization of old manifests).
- `RagIndexCache.IsValid`, `Save`, `TrySave` signatures unchanged.
- `RagSearch.BuildAsync` calls unchanged.
- Hidden tests may construct `Manifest` positionally; new fields appended with
  defaults to preserve compatibility.

## GitHub issues log
- (none yet)

## Planned commits
1. docs: plan #922 in TODO.md
2. feat: add content hash + schema version to RagIndexCache manifest
3. test: cover content-hash and schema-version invalidation
4. docs: remove TODO.md — plan executed
