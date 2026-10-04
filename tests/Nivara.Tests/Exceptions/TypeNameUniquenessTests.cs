using System.Reflection;
using NUnit.Framework;

namespace Nivara.Tests.Exceptions;

/// <summary>
/// Guards against two public types sharing a short name across different namespaces.
/// </summary>
/// <remarks>
/// <para>
/// CS0104 ("ambiguous reference between 'A.Name' and 'B.Name'") is not reachable from a
/// test that only exercises behaviour: it fires at the call site, in the consumer's file,
/// when that consumer happens to import both namespaces. Every consumer that both runs a
/// query and reads a file imports <c>Nivara.Exceptions</c> and <c>Nivara.IO</c>, so the
/// clash is the normal case rather than an edge case.
/// </para>
/// <para>
/// #532 was two public types both named <c>SchemaValidationException</c>, one in each of
/// those namespaces. It survived a documentation-gate fix (#539) that qualified the doc
/// snippet instead of the library: qualifying the snippet silences the gate without
/// changing the type, because the gate reports what the document claims compiles, not what
/// the library makes ambiguous.
/// </para>
/// <para>
/// <b>Coverage.</b> Type <i>names</i> in the two shipped assemblies only — <c>Nivara</c>
/// and <c>Nivara.Extensions</c>. Grouping by <see cref="Type.Name"/> means generic arity
/// participates, so <c>NivaraColumn</c> and <c>NivaraColumn&lt;T&gt;</c> are correctly
/// <i>not</i> flagged: arity pairs are legal C# and this gate must not fail on them.
/// </para>
/// <para>
/// That is a real limit rather than an oversight. The non-generic <c>NivaraColumn</c>
/// factory once shared a name with <c>NivaraColumn&lt;T&gt;</c> in the same namespace and
/// was renamed to <c>NivaraColumnFactory</c> by hand — this gate would not have caught
/// it, because the collision was within one namespace. Not covered: member and method
/// names, and the <c>samples/</c> assemblies (not shipped, and sample-local helpers would
/// produce false positives).
/// </para>
/// </remarks>
[TestFixture]
public class TypeNameUniquenessTests
{
    /// <summary>
    /// The two assemblies that ship to NuGet. A collision between them is just as reachable
    /// by a consumer as one within either, so they are scanned together rather than
    /// separately.
    /// </summary>
    static readonly Assembly[] ShippedAssemblies =
    [
        typeof(NivaraFrame).Assembly,
        typeof(Nivara.IO.NivaraParquetReader).Assembly,
    ];

    /// <summary>
    /// Returns every name declared in more than one namespace, as <c>Name: nsA | nsB</c>.
    /// </summary>
    /// <remarks>
    /// Kept separate from <see cref="NoShortNameIsDeclaredInTwoNamespaces"/> so the negative
    /// control can drive it with a collision that genuinely exists. A gate that only ever runs
    /// over the real assemblies can pass because reflection silently found nothing — the exact
    /// failure mode <c>NegativeControl_TheGateStillRejectsABrokenSnippet</c> exists to prevent
    /// in the documentation gate.
    /// </remarks>
    internal static IReadOnlyList<string> FindCollisions(IEnumerable<Type> types)
    {
        ArgumentNullException.ThrowIfNull(types);

        return types
            .Where(t => t is { IsNested: false, Namespace: not null })
            .GroupBy(t => t.Name, StringComparer.Ordinal)
            .Where(g => g.Select(t => t.Namespace).Distinct(StringComparer.Ordinal).Count() > 1)
            .OrderBy(g => g.Key, StringComparer.Ordinal)
            .Select(g => $"{g.Key}: {string.Join(" | ", g.Select(t => t.Namespace).Distinct(StringComparer.Ordinal).OrderBy(n => n, StringComparer.Ordinal))}")
            .ToArray();
    }

    [Test]
    public void NoShortNameIsDeclaredInTwoNamespaces()
    {
        var collisions = FindCollisions(ShippedAssemblies.SelectMany(CollectTypes));

        Assert.That(collisions, Is.Empty,
            "these type names are public in more than one namespace, so any consumer importing "
            + "both gets CS0104 at every use site. Rename one so the name carries its origin, "
            + "as #532 did for the two SchemaValidationException types: "
            + Environment.NewLine + string.Join(Environment.NewLine, collisions));
    }

    [Test]
    public void NegativeControl_TheScanReportsACollisionThatExists()
    {
        var colliding = new[] { typeof(GateProbe.Alpha.DuplicateName), typeof(GateProbe.Beta.DuplicateName) };

        // Precondition: the scan is handed two distinct types that really do share a short name
        // across two namespaces. Without this the assertion below could pass vacuously.
        Assert.Multiple(() =>
        {
            Assert.That(colliding, Has.Length.EqualTo(2));
            Assert.That(colliding[0].Name, Is.EqualTo(colliding[1].Name));
            Assert.That(colliding[0].Namespace, Is.Not.EqualTo(colliding[1].Namespace));
        });

        Assert.That(FindCollisions(colliding), Has.Exactly(1).Contains("DuplicateName"),
            "the scan failed to report a collision that is present, so a green main gate means nothing");
    }

    [Test]
    public void NegativeControl_TheScanIgnoresDistinctNames()
    {
        var distinct = new[] { typeof(NivaraFrame), typeof(NivaraColumn<int>), typeof(NivaraColumnFactory) };

        Assert.That(FindCollisions(distinct), Is.Empty,
            "differently-named types must not be reported as colliding");
    }

    static IEnumerable<Type> CollectTypes(Assembly assembly)
    {
        try
        {
            return assembly.GetTypes();
        }
        catch (ReflectionTypeLoadException ex)
        {
            return ex.Types.Where(t => t is not null)!;
        }
    }
}