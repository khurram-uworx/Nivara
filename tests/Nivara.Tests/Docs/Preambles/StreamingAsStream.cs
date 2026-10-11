namespace Nivara.Tests.Docs.Preambles.StreamingAsStream;

/// <summary>
/// Sink for the per-chunk summary call the streaming example makes. Present as a real method so the
/// snippet's argument types are checked rather than accepted on trust.
/// </summary>
public static class SnippetReport
{
    public static void Report(double sum, int rowCount) { }
}
