namespace Nivara.Tests.Docs.Preambles.StreamingFluxBasic;

/// <summary>Compile-time stand-in for the chunk sink the streaming snippets write to.</summary>
public interface ISink
{
    Task WriteAsync(NivaraFrame chunk);
}

/// <summary>
/// Stand-in for the sink the streaming snippets write to. It must stay <c>public</c>: the generated
/// snippet assembly is not named <c>Nivara.Tests</c>, so it holds no <c>InternalsVisibleTo</c> grant
/// and an internal stub would fail every snippet that uses it.
/// </summary>
public sealed class RecordingSink : ISink
{
    public Task WriteAsync(NivaraFrame chunk) => Task.CompletedTask;
}
