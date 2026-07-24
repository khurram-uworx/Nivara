# Agent Framework Workflow Patterns

Lessons learned from `samples/NivaraChat/`. Agent Framework is external; this doc captures only Nivara-specific integration notes.

## Key patterns

- `Executor<TInput, TOutput>` with `public override` — return value auto-sends downstream
- `.WithOutputFrom()` on `WorkflowBuilder` — registers executors as output sources
- Read `run.NewEvents` for `ExecutorCompletedEvent` (executor output) and `AgentResponseEvent` (LLM output)
- `OllamaApiClient` constructor doesn't throw — actual connection on `GetResponseAsync`
- **Workflow objects are single-use per run.** Do not reuse a `Workflow` instance across multiple `InProcessExecution.RunAsync` calls. Create a fresh workflow from the builder for each run (use a factory function / lambda). See [State Isolation](https://learn.microsoft.com/agent-framework/workflows/state#state-isolation).
- **Streaming output** arrives as `AgentResponseUpdateEvent` with one token per event. Accumulate per-executor-ID, then flush on `ExecutorCompletedEvent` or after all events to avoid printing each token on its own line.

## References

- `samples/NivaraChat/` — working example
- [Microsoft Agent Framework docs](https://learn.microsoft.com/agent-framework/workflows/executors)
