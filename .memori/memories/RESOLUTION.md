# Memories

- Resolved Gap 18: Added Execute_DiagnosticsRecordOptimization_WhenOptimizerChangesPlan and Execute_DiagnosticsRecordWarning_OnFailure tests to ExecutionEngineTests.cs. Custom PlanModifyingOptimizationRule always returns a new QueryPlan instance. All 1250 tests pass. Gap 18 removed from KNOWN-ISSUES.md. <!-- id=1abe918b506d4d73a45a364cd4bbecf2 entity=default type=resolution ts=2026-06-09T15:24:49.2191745+00:00 v=1 tags=phase4,gap18,resolved,execution-diagnostics,tests -->
