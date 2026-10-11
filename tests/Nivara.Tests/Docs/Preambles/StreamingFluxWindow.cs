namespace Nivara.Tests.Docs.Preambles.StreamingFluxWindow;

/// <summary>Compile-time stand-in for the pager the event-time windowing snippet pings.</summary>
public interface IPager
{
    void Ping(NivaraFrame frame);
}

/// <summary>
/// Stand-in for the pager. It must stay <c>public</c>: the generated snippet assembly is not named
/// <c>Nivara.Tests</c>, so it holds no <c>InternalsVisibleTo</c> grant and an internal stub would
/// fail every snippet that uses it.
/// </summary>
public sealed class RecordingPager : IPager
{
    public void Ping(NivaraFrame frame) { }
}
