using Nivara;

namespace Nivara.Tests.Docs.Preambles;

/// <summary>Compile-time stand-in for the chunk sink the streaming snippets write to.</summary>
public interface ISink
{
    Task WriteAsync(NivaraFrame chunk);
}

/// <summary>Compile-time stand-in for the pager the event-time windowing snippet pings.</summary>
public interface IPager
{
    void Ping(NivaraFrame frame);
}

/// <summary>
/// Stand-ins for the sinks the streaming snippets write to. They must stay <c>public</c>: the
/// generated snippet assembly is not named <c>Nivara.Tests</c>, so it holds no
/// <c>InternalsVisibleTo</c> grant and an internal stub would fail every snippet that uses it.
/// </summary>
public sealed class RecordingSink : ISink
{
    public Task WriteAsync(NivaraFrame chunk) => Task.CompletedTask;
}

/// <summary>Stand-in for the pager; public for the reason given on <see cref="RecordingSink"/>.</summary>
public sealed class RecordingPager : IPager
{
    public void Ping(NivaraFrame frame) { }
}

public sealed class Dashboard
{
    public void Update(NivaraFrame chunk) { }
}

public sealed class Archival
{
    public void Write(NivaraFrame chunk) { }
}

public sealed class LiveUi
{
    public void Push(NivaraFrame chunk) { }
}

/// <summary>
/// Sink for the per-chunk summary call the streaming example makes. Present as a real method so the
/// snippet's argument types are checked rather than accepted on trust.
/// </summary>
public static class SnippetReport
{
    public static void Report(double sum, int rowCount) { }
}