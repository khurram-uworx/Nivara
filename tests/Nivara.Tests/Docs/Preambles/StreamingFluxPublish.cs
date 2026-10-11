namespace Nivara.Tests.Docs.Preambles.StreamingFluxPublish;

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
