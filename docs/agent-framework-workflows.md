# Agent Framework Workflow Patterns

Lessons learned from `samples/NivaraChat/`. Agent Framework is external; this doc captures only Nivara-specific integration notes.

## Key patterns

- `Executor<TInput, TOutput>` with `public override` — return value auto-sends downstream
- `.WithOutputFrom()` on `WorkflowBuilder` — registers executors as output sources
- Read `run.NewEvents` for `ExecutorCompletedEvent` (executor output) and `AgentResponseEvent` (LLM output)
- `OllamaApiClient` constructor doesn't throw — actual connection on `GetResponseAsync`

## References

- `samples/NivaraChat/` — working example
- [Microsoft Agent Framework docs](https://learn.microsoft.com/agent-framework/workflows/executors)
