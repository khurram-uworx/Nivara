# Memories

- Phase 1 review completed: Scope creep found: CanReadInChunks/EstimatedRowCount/ReadChunkAsync added to IQuerySource without plan coverage. Dead code: calculateChunkSize result discarded in StreamingExecutionStrategy. EagerExecutionStrategy.ExecuteCoreAsync has redundant per-operation Task.Run overhead. ParallelExecutionStrategy casts IQueryOperation to concrete types (SortOperation, GroupByOperation) violating LSP. <!-- id=150c4dd1016044f0b1f9a02bad84ffa5 entity=default type=fact ts=2026-06-09T10:25:10.5777720+00:00 v=1 tags=phase1,review,gaps,pascalcase,naming,conventions -->
