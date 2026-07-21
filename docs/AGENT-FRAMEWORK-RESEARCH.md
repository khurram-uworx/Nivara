# Agent Framework Research — Microsoft.Extensions.AI + Microsoft.Agents.AI.Workflows

## Purpose

This document captures the research done in July 2026 to determine how a
Nivara-trained model can participate in the standard .NET AI ecosystem as a
first-class component. It records API signatures, architectural patterns,
and design decisions so that future iterations do not need to rediscover
this information.

## Relevant Packages

| Package | Version | Purpose |
|---------|---------|---------|
| `Microsoft.Extensions.AI.Abstractions` | 10.7.0 | `IChatClient`, `ChatMessage`, `ChatResponse`, `ChatOptions` |
| `Microsoft.Extensions.AI` | 10.7.0 | `FunctionInvokingChatClient`, `ChatClientBuilder`, `DistributedCache` |
| `Microsoft.Agents.AI.Abstractions` | 1.13.0 | `AIAgent`, `ChatClientAgent` |
| `Microsoft.Agents.AI` | 1.13.0 | `ChatClientAgent` implementation |
| `Microsoft.Agents.AI.Workflows` | 1.13.0 | `WorkflowBuilder`, `Executor`, `[MessageHandler]`, `AgentWorkflowBuilder` |

## Three Levels of Integration

### Level 1: AIFunction Tool (canonical, simplest)

A Nivara model wrapped as an `AIFunction` that the LLM calls when needed.

```csharp
// Nivara model logic as a tool
var classifyIntent = AIFunctionFactory.Create(
    (string text) => new NivaraClassifier().Classify(text),
    "classify_intent",
    "Returns the intent category of a user message: support, billing, general");

// Registered per-request
var options = new ChatOptions
{
    Tools = [ classifyIntent ]
};
```

**Limitation:** The LLM decides when to call the tool. For deterministic
gatekeeping (always classify before routing), the function-calling loop
may not fire every time.

### Level 2: IChatClient + DelegatingChatClient Middleware

A Nivara model implements `IChatClient` directly. It does not call an LLM;
it returns deterministic results (classification, scores, entity lists).
The `ChatClientAgent` wrapper makes it an `AIAgent` for DI and pipeline
use.

```csharp
public sealed class NivaraChatClient : IChatClient
{
    // Override GetResponseAsync, GetStreamingResponseAsync, GetService
}

// Usage via ChatClientAgent
var agent = new ChatClientAgent(
    new NivaraChatClient(model, tokenizer),
    instructions: "Classify input and return structured output.");

// Or as IChatClient middleware in a pipeline
IChatClient pipeline = new ChatClientBuilder(llmClient)
    .Use(inner => new NivaraPreprocessingChatClient(inner))
    .UseFunctionInvocation()
    .Build();
```

### Level 3: Custom Executor in a Workflow (the killer pattern — RECOMMENDED)

A Nivara model as a custom `Executor` subclass in a `WorkflowBuilder` graph,
alongside `AIAgent` nodes. This is the most natural .NET Agent Framework
pattern: deterministic computation nodes and LLM-backed agent nodes coexist
in one directed graph.

```csharp
// Custom executor: Nivara model does deterministic classification
internal sealed partial class NivaraSentimentExecutor : Executor
{
    private readonly NivaraClassifier _model;

    public NivaraSentimentExecutor(NivaraClassifier model)
        : base("NivaraSentiment") => _model = model;

    [MessageHandler]
    public ValueTask<string> HandleAsync(string text, IWorkflowContext ctx)
    {
        var result = _model.Classify(text);
        return ValueTask.FromResult(result);
    }
}

// LLM agent
var llmAgent = new ChatClientAgent(
    openAIChatClient, instructions: "You are a helpful assistant.");

// Workflow graph
var workflow = new WorkflowBuilder(new NivaraSentimentExecutor(model))
    .AddEdge("NivaraSentiment", llmAgent)   // route classified text to LLM
    .Build();
```

## IChatClient API Reference

### Key Interface

```csharp
public interface IChatClient : IDisposable
{
    Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default);

    IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default);

    object? GetService(Type serviceType, object? serviceKey);
}
```

### DelegatingChatClient Base

```csharp
public class DelegatingChatClient : IChatClient
{
    protected DelegatingChatClient(IChatClient innerClient);
    // Override GetResponseAsync / GetStreamingResponseAsync
    // Override GetService — default passes through to inner
}
```

### ChatClientBuilder

```csharp
public class ChatClientBuilder
{
    // Overload 1: full IChatClient wrapper
    public ChatClientBuilder Use(Func<IChatClient, IChatClient> clientFactory);
    // Overload 2: shared delegate for both streaming and non-streaming
    public ChatClientBuilder Use(Func<..., Func<..., Task>, ..., Task> sharedFunc);
    // Overload 3: separate delegates for response vs streaming
    public ChatClientBuilder Use(
        Func<..., IChatClient, ..., Task<ChatResponse>>? getResponseFunc,
        Func<..., IChatClient, ..., IAsyncEnumerable<ChatResponseUpdate>>? getStreamingFunc);
}
```

### FunctionInvokingChatClient

Wraps an inner `IChatClient` with automatic tool-calling loop:

```csharp
public class FunctionInvokingChatClient : DelegatingChatClient
{
    public FunctionInvokingChatClient(
        IChatClient innerClient,
        ILoggerFactory? loggerFactory = null,
        IServiceProvider? functionInvocationServices = null);

    public int MaximumIterationsPerRequest { get; set; }  // default 40
    public AIFunction[]? AdditionalTools { get; set; }
    public bool AllowConcurrentInvocation { get; set; }
    // etc.
}
```

Built-in pipeline. The outer layer `FunctionInvokingChatClient` manages
the tool loop; the inner client is the LLM. A non-LLM IChatClient in the
inner position cannot produce `FunctionCallContent`, so the tool loop
never kicks in. But the Nivara IChatClient works fine as middleware
(before/after the LLM) or as a standalone agent that returns deterministic
results.

## Agent Framework API Reference

### ChatClientAgent

```csharp
public sealed class ChatClientAgent : AIAgent
{
    public ChatClientAgent(
        IChatClient chatClient,
        ChatClientAgentOptions? options = null);
}
```

`ChatClientAgent` wraps any `IChatClient` as an `AIAgent`. By default it
adds function-calling support (`FunctionInvokingChatClient` wrapper). Set
`UseProvidedChatClientAsIs = true` to skip that.

### AIAgent — base class

```csharp
public abstract class AIAgent
{
    public string Id { get; }
    public string? Name { get; }

    // Non-streaming
    public Task<AgentResponse> RunAsync(
        string input, AgentRunOptions? options = null,
        CancellationToken ct = default);

    // Streaming
    public IAsyncEnumerable<AgentResponseUpdate> RunStreamingAsync(
        string input, AgentRunOptions? options = null,
        CancellationToken ct = default);
}
```

### Agent Middleware Pipeline

`ChatClientAgent` has three middleware layers:
1. **Agent middleware** — wraps `RunAsync`/`RunStreamingAsync` via `.AsBuilder().Use()`
2. **Context layer** — `ChatHistoryProvider` + `AIContextProviders`
3. **IChatClient middleware** — `ChatClientBuilder.Use()` decorations

### WorkflowBuilder

```csharp
public class WorkflowBuilder
{
    // Constructor takes the start executor
    public WorkflowBuilder(Executor start);
    public WorkflowBuilder(ExecutorBinding start);

    // Overloads accepting AIAgent directly (auto-wraps to AIAgentBinding)
    public WorkflowBuilder(AIAgent start);

    public WorkflowBuilder AddEdge(ExecutorBinding source, ExecutorBinding target);
    public WorkflowBuilder AddEdge<T>(
        ExecutorBinding source, ExecutorBinding target,
        Func<T?, bool>? condition = null);

    public WorkflowBuilder AddFanOutEdge(
        ExecutorBinding source, IEnumerable<ExecutorBinding> targets);

    public Workflow Build();
}
```

Both `Executor` subclasses and `AIAgent` instances can be connected in
the same graph. All input/output routing is type-safe via the
`[MessageHandler]` signature.

### Custom Executor

```csharp
internal sealed partial class MyExecutor() : Executor("MyExecutor")
{
    [MessageHandler]
    private ValueTask<string> HandleAsync(string message, IWorkflowContext context)
    {
        // Do deterministic work
        return ValueTask.FromResult(result);
    }
}
```

Source generation handles handler registration. The class must be `partial`.
Return type determines output routing — the value is automatically sent to
downstream executors. Alternatively, use `context.SendMessageAsync(result)`
to send manually.

### AgentWorkflowBuilder (convenience helpers)

```csharp
public static class AgentWorkflowBuilder
{
    public static Workflow BuildSequential(params AIAgent[] agents);
    public static Workflow BuildConcurrent(params AIAgent[] agents);
    public static GroupChatBuilder CreateGroupChatBuilderWith(
        Func<IReadOnlyList<AIAgent>, IGroupChatManager> managerFactory);
}
```

These only work with `AIAgent` instances (not custom `Executor` instances).
For workflows mixing both, use `WorkflowBuilder` directly.

### ExecutorInstanceBinding

```csharp
// Wraps an Executor instance for use in WorkflowBuilder
var binding = new ExecutorInstanceBinding(myExecutor);
builder.AddEdge(binding, otherExecutor);
```

### AIAgentBinding

```csharp
public class AIAgentBinding : ExecutorBinding
{
    public AIAgentBinding(AIAgent agent, AIAgentHostOptions? options = null);
}
```

Automatically created when you pass an `AIAgent` to `WorkflowBuilder` or
`AddEdge`.

### ExecutorBindingExtensions.BindAsExecutor

For function-based executors (no subclass needed):

```csharp
// Synchronous handler
Func<string, string> handler = text => text.ToUpper();
var binding = handler.BindAsExecutor("uppercase");

// Async handler
Func<string, ValueTask<string>> asyncHandler = async text => await DoWork(text);
var binding = asyncHandler.BindAsExecutor("async-worker");
```

Both produce `ExecutorBinding` instances that the `WorkflowBuilder`
accepts. Supports `IWorkflowContext` and `CancellationToken` overloads.

## Execution Model: Supersteps

The workflow engine uses a modified Pregel (Bulk Synchronous Parallel) model:

1. Each superstep collects all pending messages
2. Routes them to target executors based on edge definitions
3. Runs all triggered executors concurrently within the superstep
4. Waits for all to complete (synchronization barrier)
5. Queues new messages for the next superstep

## Key Design Decision

**The recommended integration path for Nivara is the custom `Executor`
subclass (Level 3), not the standalone IChatClient (Level 2).** Reasons:

1. `Executor` is the first-class way to represent deterministic computation
   in the Agent Framework workflow graph.
2. `[MessageHandler]` source-gen gives type-safe routing and compile-time
   validation.
3. Nivara executors can be mixed with AIAgent nodes in the same graph.
4. The IChatClient approach (`ChatClientAgent`) adds unnecessary overhead
   (function-calling wrapper, turn-token protocol) when the model is not
   an LLM.

However, the IChatClient approach may still be useful as a **complementary
pattern** — e.g., NivaraChatClient as middleware in a ChatClientBuilder
pipeline for pre/post processing. Both paths are viable; the Executor path
is showcased as primary.

## Agent Framework Workflows Scaffold

```bash
dotnet add package Microsoft.Agents.AI.Workflows --version 1.13.0
dotnet add package Microsoft.Agents.AI --version 1.13.0
dotnet add package Microsoft.Extensions.AI --version 10.7.0
dotnet add package Microsoft.Extensions.AI.Abstractions --version 10.7.0
```

For the LLM node:
```bash
dotnet add package Microsoft.Extensions.AI.OpenAI --version 10.7.0
```

## Cross-reference

- IChatClient middleware docs: `learn.microsoft.com/dotnet/ai/ichatclient`
- Agent Framework workflows: `learn.microsoft.com/agent-framework/workflows`
- Agent Framework executors: `learn.microsoft.com/agent-framework/workflows/executors`
- Agent Framework agents: `learn.microsoft.com/agent-framework/agents`
