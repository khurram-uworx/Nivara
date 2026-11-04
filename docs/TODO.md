# Plan: Improve --rag to use Nivara convention (#912)

Issue: https://github.com/khurram-uworx/Nivara/issues/912
Branch: khurram/912

## Problem

`--rag` and `--rag-agent` currently use `InMemoryVectorStore` + `DocumentChunk` (parallel
arrays / vector store). They do not exercise the #867 embedding convention
(`NivaraFrame` + `NivaraColumn<float[]>` + `ToEmbeddingMatrix` + `RowCosineSimilarity` +
`TopKDescending`), null semantics, ACL filtering, citations, round-trip, or Silk E2E.
Issue #912 (and dependents #914/#915) need the convention demonstrated on a real corpus.

## Proposed changes

### A. Shared Silk+CPU MiniLM embedder (samples/Nivara.Samples/MiniLMEmbedder.cs)
`Create(modelDir, useSilk)` -> `(Func<string,float[]> Embed, int Dimension, IDisposable Lifetime)`.
CPU path wraps `MiniLMEmbeddingGenerator.Create`; Silk path ports OpenSilkEncoder pattern.

### B. Chunker (Helpers/DocumentChunk.cs)
Add `RagChunk` record + `ChunkMarkdown` producing ChunkId, Source, Title, SectionPath,
AclGroup (parent dir name lowercased), Content. Empty content -> null embedding.

### C. --rag rewrite (Modes/RagMode.cs)
Store selector (frame default, inmemory legacy). Frame path builds NivaraFrame, ACL filter,
ToEmbeddingMatrix, RowCosineSimilarity, TopKDescending. Print chunks w/ metadata.
Ollama optional. --roundtrip flag.

### D. --rag-agent migration (Modes/RagAgentMode.cs)
Same indexing; TextSearchProvider delegate uses frame retrieval for frame store.

### E. CLI (Program.cs, ModeContext)
Add UseSilk, AclGroup, ShowRoundTrip, Store (defaults). Parse --silk, --acl-group, --roundtrip, --store.

### F. Docs/README updates.

## Verification
- dotnet build Nivara.slnx
- targeted tests (Nivara.Tests AI + Docs)
- manual smoke runs

## Planned commits
1. docs: plan in TODO.md
2. feat(samples): shared MiniLMEmbedder
3. feat(NivaraChat): RagChunk + chunker
4. feat(NivaraChat): --rag frame backend + flags + optional ollama
5. feat(NivaraChat): --rag-agent migration
6. docs: README + flags
7. (fixes as needed)

## GitHub issues log
- (none yet)

## Blast radius
- NivaraChat Modes, Helpers, Program.cs, ModeContext
- Nivara.Samples (new file)
- docs/README.md
- tests: none modified unless needed
