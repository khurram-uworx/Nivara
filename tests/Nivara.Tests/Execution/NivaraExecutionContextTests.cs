using Nivara.Diagnostics;
using Nivara.Execution;
using NUnit.Framework;

namespace Nivara.Tests.Execution;

[TestFixture]
public class NivaraExecutionContextTests
{
    [Test]
    public void DiagnosticsProperty_DefaultNull()
    {
        var context = new NivaraExecutionContext();
        Assert.That(context.ExecutionDiagnostics, Is.Null);
    }

    [Test]
    public void DiagnosticsProperty_SetGetRoundTrip()
    {
        var context = new NivaraExecutionContext();
        var diagnostics = new ExecutionDiagnostics();
        context.ExecutionDiagnostics = diagnostics;
        Assert.That(context.ExecutionDiagnostics, Is.SameAs(diagnostics));
    }

    [Test]
    public void Clone_CopiesDiagnostics()
    {
        var context = new NivaraExecutionContext();
        var diagnostics = new ExecutionDiagnostics();
        context.ExecutionDiagnostics = diagnostics;
        var clone = context.Clone();
        Assert.That(clone.ExecutionDiagnostics, Is.SameAs(diagnostics));
    }

    [Test]
    public void ToString_IncludesDiagnosticsStatus()
    {
        var context = new NivaraExecutionContext();
        Assert.That(context.ToString(), Does.Contain("Diagnostics: None"));
        context.ExecutionDiagnostics = new ExecutionDiagnostics();
        Assert.That(context.ToString(), Does.Contain("Diagnostics: Set"));
    }
}
