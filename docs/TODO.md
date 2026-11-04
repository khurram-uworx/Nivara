# Plan: Cache RAG index as Parquet dimension frame (#912 follow-up)

## Problem
Each --rag launch re-chunks and re-embeds the whole corpus (~45s CPU). Cache the ingested
index so subsequent launches load in seconds. Parquet can't store float[] columns, so cache
the dimension frame (E0..E{dim-1}) + metadata instead.

## Design
- Cache identity = hash(resolvedDocsDir + "|" + backend) -> samples/data/nivara/rag_cache/rag_index_<hash>.parquet
- --index <path> overrides; --rebuild forces; --no-cache disables.
- Manifest in Parquet file metadata: docsMaxMtime, chunkCount, backend, dimension.
- Invalidation: manifest mismatch or any source doc newer than docsMaxMtime or chunk count mismatch.
- Load: read Parquet -> dimension frame; reconstruct vector column via ToEmbeddingFrame if needed.
- Score on dimension frame with core RowCosineSimilarity(querySeries, labels).
- --rag-agent shares the cache.

## Planned commits
1. docs: plan in TODO.md
2. feat: cache load/save in RagMode
3. feat: cache in RagAgentMode
4. feat: CLI flags --index/--rebuild/--no-cache
5. docs: README
6. remove TODO.md

## GitHub issues log
- (none)
