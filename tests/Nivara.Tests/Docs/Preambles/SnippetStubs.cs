using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Operations;

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

/// <summary>
/// Row types the typed-query snippets bind against. <c>Query&lt;T&gt;()</c> maps each property to a
/// like-named column case-insensitively, so what the gate actually checks here is that the predicate
/// and projection expressions reference members that exist and that the chain resolves — the
/// property-to-column mapping itself is a runtime <c>QuerySchemaValidationException</c>, never a compile
/// error, so it was never something this gate verified.
/// </summary>
/// <remarks>
/// <para>
/// <b>Known shape conflict.</b> Two different <c>Employee</c> rows exist in the documentation:
/// <c>EXAMPLES.md</c> §5a is <c>{Name, Department, Salary, IsActive}</c> over an <c>int</c> Salary
/// column, while §5c is <c>{Name, City, Age, Salary}</c> over a <c>double</c> Salary column. §5b
/// then declares a third, deliberately wrong one (string Salary) to demonstrate eager schema
/// validation. Preamble stubs share one global namespace and cannot be overloaded by document, so
/// this is the union and <c>Salary</c> is <c>double</c> — which satisfies both sets of expressions,
/// since §5a only ever orders and projects on it. That the union matches no real frame is recorded
/// rather than hidden; see issue #524.
/// </para>
/// <para>
/// §5b is unaffected: it declares its own <c>Employee</c> in the global namespace of its block, and a
/// type in the compilation unit's own namespace wins over one merely imported from here.
/// </para>
/// </remarks>
public sealed class Employee
{
    public string Name { get; set; } = string.Empty;
    public string Department { get; set; } = string.Empty;
    public string City { get; set; } = string.Empty;
    public int Age { get; set; }
    public double Salary { get; set; }
    public bool IsActive { get; set; }
}

/// <summary>Row type for the risk-scoring query, matching the <c>CustomerId/Segment/RiskScore</c> columns.</summary>
public sealed class Customer
{
    public string CustomerId { get; set; } = string.Empty;
    public string Segment { get; set; } = string.Empty;
    public float RiskScore { get; set; }
}

/// <summary>
/// Row type for the typed-query sections, the union of the two <c>Person</c> rows the document
/// declares: <c>GETTING-STARTED.md</c> "Basic Queries" writes <c>{Name, Age, Salary}</c> while
/// "Typed Object LINQ" writes <c>{Name, Department, Age, Salary}</c> and groups by
/// <c>Department</c>. One stub cannot be both, and the union is what satisfies every expression.
/// <see cref="Employee"/> carries the same caveat for the same reason.
/// </summary>
public sealed class Person
{
    public string Name { get; set; } = string.Empty;
    public string Department { get; set; } = string.Empty;
    public int Age { get; set; }
    public double Salary { get; set; }
}

/// <summary>
/// Row type for the null-ordering example. <c>Score</c> is <c>int</c> rather than <c>int?</c> because
/// the document's own <c>Player</c> declaration is not what this resolves for — "Null Handling in
/// Sorting" declares its own in a <c>mode: File</c> block, and a type in the compilation unit's own
/// namespace wins over one merely imported from here. This stub serves the blocks that do not
/// declare their own.
/// </summary>
public sealed class Player
{
    public string Name { get; set; } = string.Empty;
    public int Score { get; set; }
}

/// <summary>
/// Row type for the fluent-API and execution examples, matching their <c>Name</c>, <c>Age</c> and
/// <c>Score</c> columns. <c>Score</c> is <c>double</c> here, unlike <see cref="Player"/>'s
/// <c>int</c>, because the two sections build their frames from different column types.
/// </summary>
public sealed class Contestant
{
    public string Name { get; set; } = string.Empty;
    public int Age { get; set; }
    public double Score { get; set; }
}

/// <summary>Row type for the JSON scanning example, matching the <c>Id/Name/Email/Active</c> fields.</summary>
public sealed class User
{
    public string Id { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Email { get; set; } = string.Empty;
    public bool Active { get; set; }
}

/// <summary>
/// Compile-time stand-in for the <c>Module&lt;float&gt;</c> subclasses the AutoDiff snippets declare in
/// their own <c>mode: File</c> block. Mirrors that declaration member for member so the training-loop
/// snippets bind against the same surface — <c>Forward</c> is overridden rather than stubbed out,
/// because a snippet that passes the tensor through a <c>Linear</c> layer must have that call checked.
/// </summary>
public class LinearModel : Module<float>
{
    Linear<float> L1;

    public LinearModel()
    {
        L1 = new Linear<float>(3, 1);
        RegisterModules(L1);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> x)
        => L1.Forward(x);
}

/// <summary>Compile-time stand-in for the Act 8 fraud classifier, mirroring its declared shape.</summary>
public class FraudNet : Module<float>
{
    Linear<float> L1, L2, L3;

    public FraudNet()
    {
        L1 = new Linear<float>(8, 64);
        L2 = new Linear<float>(64, 32);
        L3 = new Linear<float>(32, 1);
        RegisterModules(L1, L2, L3);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> x)
    {
        var h = ReverseGradOperations.Relu(L1.Forward(x));
        h = ReverseGradOperations.Relu(L2.Forward(h));
        return L3.Forward(h);
    }
}