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

sealed class RecordingSink : ISink
{
    public Task WriteAsync(NivaraFrame chunk) => Task.CompletedTask;
}

sealed class RecordingPager : IPager
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