using Nivara.IO;
using Nivara.Samples.Incident;
using NUnit.Framework;
using System.Reflection;

namespace Nivara.Tests.Incident;

/// <summary>
/// Gates that every public member of <c>Nivara.Samples.Incident</c> is accounted for, and
/// that the ones which produce a result are executed and inspected.
/// <para>
/// <see cref="Analysis.AnalyzeGroupedAggregationWithTypedLinq"/> is why that matters. It threw
/// <c>SchemaValidationException</c> on first call: it mapped <c>RequestRow</c>, which declares
/// <c>DurationPercentRank</c>, but its pipeline never created that column. #496 fixed the method,
/// and the file-handle probe added then did invoke it &#8212; but that probe only checked the
/// handle, so no test asserted anything about what the method returned. A method can be called
/// and still be unverified.
/// </para>
/// <para>
/// <b>Why this is an execution gate and not a coverage-presence gate.</b> A gate that only asked
/// "is this member named in some test?" would not have caught that method either &#8212;
/// <c>AnalysisResourceTests</c> already named it and passed, because it never looked at a result.
/// So every row here is actually <em>run</em> and its output inspected. Handle release is a
/// separate concern and is gated by #498's <c>FileHandleProbe</c>.
/// </para>
/// <para>
/// <b>Row granularity.</b> Rows are the public callable members &#8212; methods, static
/// properties, static fields &#8212; plus one row per public type. Instance properties on the
/// data types are not separate rows: <see cref="RequestRow"/> and <see cref="InstanceRow"/> are
/// covered transitively by the analyses that call <c>Query&lt;T&gt;()</c> on them, and that
/// binding is precisely the operation that threw the original
/// <c>SchemaValidationException</c>, so the row is bound tightly to the bug class.
/// </para>
/// <para>
/// <para>
/// <b>What this gate does not claim.</b> For members that return a frame or a summary, it proves
/// the call completed and the result is non-empty; it does not prove the values are correct
/// &#8212; that stays with the per-fixture assertions in <see cref="AnalysisTests"/> and
/// <see cref="StreamixScenarioTests"/>. The 2 covered-by and 5 static-holder rows are resolved
/// without being executed here: the holders by their members' exercises, and the two data types
/// by the <c>Query&lt;T&gt;()</c> binding performed by a named exercise, which is itself asserted
/// by that exercise running. The <see cref="ServiceEvent"/> and <see cref="IncidentScenario"/>
/// rows are the weakest &#8212; they confirm construction and assignment, and exist mainly so the
/// types are not silently untested. Nested summary records are excluded because they are
/// constructed by the registered <c>Run*</c> exercises.
/// </para>
/// <para>
/// <b>Known limit of the covered-by mechanism.</b> Nothing here verifies that the named exercise
/// still binds the type it is credited with covering. If an analysis stopped calling
/// <c>Query&lt;RequestRow&gt;()</c>, the <c>RequestRow</c> row would stay resolved and its
/// properties would rot undetected &#8212; which is the bug class this file exists to prevent. The
/// binding is not introspectable from here, so the row states the intent rather than enforcing it.
/// </para>
/// </para>
/// </summary>
[TestFixture]
public class IncidentSurfaceTests
{
    const string IncidentNamespace = "Nivara.Samples.Incident";
    const int TotalRecords = 10_000;
    const int ChunkSize = 1_000;

    string tempDir = null!;

    [OneTimeSetUp]
    public void OneTimeSetUp()
    {
        tempDir = Path.Combine(Path.GetTempPath(), $"inc-surface-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        DatasetGenerator.GenerateFromRecordCount(tempDir, "A", TotalRecords);
    }

    [OneTimeTearDown]
    public void OneTimeTearDown()
    {
        if (tempDir is not null && Directory.Exists(tempDir))
            Directory.Delete(tempDir, true);
    }

    [Test]
    public void IncidentSurface_HasNoUnregisteredMembers()
    {
        var surface = Surface();
        var exercises = Exercises();
        var transitive = CoveredTransitively();
        var skipped = Skipped();

        var resolved = new HashSet<string>(exercises.Keys, StringComparer.Ordinal);
        foreach (var member in transitive.Keys) resolved.Add(member);
        foreach (var member in skipped.Keys) resolved.Add(member);

        // A static holder class (Analysis, Ingestion, ...) has no behaviour of its own, so its row
        // is satisfied when every member it declares is resolved. Spelling that out here rather
        // than hand-listing five more registry entries keeps the two from drifting apart.
        foreach (var holder in StaticHolderTypes())
        {
            var members = surface
                .Where(row => row.StartsWith(holder + ".", StringComparison.Ordinal))
                .ToList();

            if (members.Count > 0 && members.All(resolved.Contains))
                resolved.Add(holder);
        }

        var unregistered = surface
            .Where(row => !resolved.Contains(row))
            .OrderBy(row => row, StringComparer.Ordinal)
            .ToList();

        Assert.That(unregistered, Is.Empty,
            "These public members of Nivara.Samples.Incident are neither executed, nor declared "
            + "covered-by another exercise, nor skipped with a reason. That is exactly how "
            + "AnalyzeGroupedAggregationWithTypedLinq shipped broken in #499: nothing called it, so "
            + "nothing noticed it threw. Add an entry to Exercises() that runs the member and "
            + "inspects its result, or -- if it genuinely cannot run here -- to Skipped() with a "
            + "reason a reviewer can check.");

        // A covered-by row is only meaningful if the exercise it names is itself registered.
        // Otherwise a typo silently converts an unexercised member into a "covered" one.
        var dangling = transitive
            .Where(pair => !exercises.ContainsKey(pair.Value))
            .OrderBy(pair => pair.Key, StringComparer.Ordinal)
            .Select(pair => $"{pair.Key} is declared covered-by '{pair.Value}', which is not a registered exercise")
            .ToList();

        Assert.That(dangling, Is.Empty,
            "A covered-by row names an exercise that does not exist, so it is covering nothing. "
            + "If that exercise is gone, the row has lost its coverage.");

        var missingReasons = skipped.Where(pair => string.IsNullOrWhiteSpace(pair.Value)).ToList();
        Assert.That(missingReasons, Is.Empty, "Every skipped member needs a reason a reviewer can check.");

        TestContext.Out.WriteLine(
            $"Incident surface: {resolved.Count} of {surface.Count} public members resolved — "
            + $"{exercises.Count} exercised, {transitive.Count} covered-by, {skipped.Count} skipped, "
            + $"{resolved.Count - exercises.Count - transitive.Count - skipped.Count} static holder types.");
    }

    [Test]
    public async Task IncidentSurface_AllExercisesSucceed()
    {
        var exercises = Exercises();
        var failures = new List<string>();

        foreach (var (member, exercise) in exercises.OrderBy(pair => pair.Key, StringComparer.Ordinal))
        {
            try
            {
                await exercise(CancellationToken.None);
            }
            catch (Exception ex)
            {
                // One reason per member: a failure here is attributed to exactly one row, so the
                // count of failures and the count of members that ran never disagree.
                failures.Add($"{member}: {ex.GetType().Name}: {ex.Message}");
            }
        }

        TestContext.Out.WriteLine($"Incident surface: {exercises.Count - failures.Count} of {exercises.Count} members executed successfully.");

        Assert.That(failures, Is.Empty,
            "These Incident sample members threw when executed. A member that has never run is "
            + "indistinguishable from a working one, which is the gap #499 is about.");
    }

    /// <summary>
    /// The reflected public surface: every public type plus each public method, public static
    /// property, and public static field it declares.
    /// </summary>
    static IReadOnlyList<string> Surface()
    {
        const BindingFlags Declared = BindingFlags.Public | BindingFlags.Static | BindingFlags.Instance | BindingFlags.DeclaredOnly;

        var rows = new List<string>();

        var types = typeof(Analysis).Assembly
            .GetExportedTypes()
            .Where(t => t.Namespace == IncidentNamespace)
            .Where(t => !t.IsNested)
            .OrderBy(t => t.Name, StringComparer.Ordinal);

        foreach (var type in types)
        {
            rows.Add(type.Name);

            // IsSpecialName drops property accessors and operator overloads; the angle-bracket
            // filter drops compiler-generated members such as the record <Clone>$ method.
            foreach (var method in type.GetMethods(Declared)
                .Where(m => !m.IsSpecialName)
                .Where(m => !m.Name.StartsWith('<'))
                .OrderBy(m => m.Name, StringComparer.Ordinal))
            {
                rows.Add($"{type.Name}.{method.Name}");
            }

            foreach (var property in type.GetProperties(Declared)
                .Where(p => p.GetMethod is { IsStatic: true })
                .OrderBy(p => p.Name, StringComparer.Ordinal))
            {
                rows.Add($"{type.Name}.{property.Name}");
            }

            foreach (var field in type.GetFields(Declared)
                .Where(f => f.IsStatic)
                .OrderBy(f => f.Name, StringComparer.Ordinal))
            {
                rows.Add($"{type.Name}.{field.Name}");
            }
        }

        return rows;
    }

    /// <summary>
    /// Builds a surface row key from a declaring type name and a member name, so the registry
    /// cannot drift out of step with <see cref="Surface"/>. Both halves come from
    /// <c>nameof</c>, so a renamed or misspelled member is a compile error rather than a registry
    /// entry that silently matches nothing.
    /// </summary>
    static string Key(string declaringType, string member) => $"{declaringType}.{member}";

    static IEnumerable<string> StaticHolderTypes() => typeof(Analysis).Assembly
        .GetExportedTypes()
        .Where(t => t.Namespace == IncidentNamespace)
        .Where(t => !t.IsNested)
        .Where(t => t.IsAbstract && t.IsSealed)
        .Select(t => t.Name)
        .OrderBy(name => name, StringComparer.Ordinal);

    /// <summary>
    /// Types whose shape is only meaningful through a frame binding, plus the exercise that
    /// performs that binding. If the named exercise fails, test 2 goes red and this row's coverage
    /// is void with it — so the chain does not rest on the name alone.
    /// </summary>
    static Dictionary<string, string> CoveredTransitively() => new(StringComparer.Ordinal)
    {
        ["RequestRow"] = Key(nameof(Analysis), nameof(Analysis.AnalyzeRegionalPartitioning)),
        ["InstanceRow"] = Key(nameof(Analysis), nameof(Analysis.AnalyzeSaturationOrdering)),
    };

    static Dictionary<string, string> Skipped() => new(StringComparer.Ordinal)
    {
        ["DatasetGenerator.Generate"] =
            "writes 10_000_000 * scale records, so it cannot run in CI. Its row count and field "
            + "ranges are verified by tests/Nivara.PerformanceTests/IncidentLabBenchmark.cs, a "
            + "manual `dotnet run -c Release` harness that CI never invokes.",
    };

    Dictionary<string, Func<CancellationToken, Task>> Exercises()
    {
        string requestsPath = Path.Combine(tempDir, "requests.parquet");
        string csvPath = Path.Combine(tempDir, "requests.csv");
        var scenario = Scenarios.A;

        return new Dictionary<string, Func<CancellationToken, Task>>(StringComparer.Ordinal)
        {
            // --- Analysis: every entry point must run and produce rows. ---------------

            [Key(nameof(Analysis), nameof(Analysis.AnalyzeDegradationOrdering))] = _ =>
            {
                using var query = Analysis.AnalyzeDegradationOrdering(tempDir, scenario);
                using var frame = query.Collect();
                Assert.That(frame.RowCount, Is.GreaterThan(0), "expected a non-empty degradation ordering");
                return Task.CompletedTask;
            },

            [Key(nameof(Analysis), nameof(Analysis.AnalyzeDeploymentCorrelation))] = _ =>
            {
                using var frame = Analysis.AnalyzeDeploymentCorrelation(tempDir, scenario);
                Assert.That(frame.RowCount, Is.GreaterThan(0), "expected a non-empty deployment correlation");
                return Task.CompletedTask;
            },

            [Key(nameof(Analysis), nameof(Analysis.AnalyzeSaturationOrdering))] = _ =>
            {
                using var frame = Analysis.AnalyzeSaturationOrdering(tempDir, scenario);
                Assert.That(frame.RowCount, Is.GreaterThan(0), "expected a non-empty saturation ordering");
                return Task.CompletedTask;
            },

            [Key(nameof(Analysis), nameof(Analysis.AnalyzeRegionalPartitioning))] = _ =>
            {
                using var frame = Analysis.AnalyzeRegionalPartitioning(tempDir, scenario);
                Assert.That(frame.RowCount, Is.GreaterThan(0), "expected a non-empty regional partitioning");
                return Task.CompletedTask;
            },

            [Key(nameof(Analysis), nameof(Analysis.AnalyzeGroupedAggregation))] = _ =>
            {
                using var frame = Analysis.AnalyzeGroupedAggregation(tempDir, scenario);
                Assert.That(frame.RowCount, Is.GreaterThan(0), "expected a non-empty grouped aggregation");
                return Task.CompletedTask;
            },

            [Key(nameof(Analysis), nameof(Analysis.AnalyzeGroupedAggregationWithTypedLinq))] = _ =>
            {
                using var frame = Analysis.AnalyzeGroupedAggregationWithTypedLinq(tempDir, scenario);
                Assert.That(frame.RowCount, Is.GreaterThan(0), "expected a non-empty typed LINQ grouped aggregation");
                return Task.CompletedTask;
            },

            // --- DatasetGenerator --------------------------------------------------
            // The fixture in OneTimeSetUp generates through this method, so asserting its
            // outputs exist is an assertion about the method, not about the fixture.
            [Key(nameof(DatasetGenerator), nameof(DatasetGenerator.GenerateFromRecordCount))] = _ =>
            {
                foreach (var file in new[] { "requests.parquet", "requests.csv", "deployments.parquet", "instances.parquet" })
                {
                    var path = Path.Combine(tempDir, file);
                    Assert.That(File.Exists(path), Is.True, $"OneTimeSetUp generated through GenerateFromRecordCount but '{file}' is missing");
                }
                return Task.CompletedTask;
            },

            // --- Ingestion ---------------------------------------------------------
            [Key(nameof(Ingestion), nameof(Ingestion.LoadParquet))] = _ =>
            {
                using var query = Ingestion.LoadParquet(requestsPath);
                using var frame = query.Collect();
                Assert.That(frame.RowCount, Is.GreaterThan(0), "expected LoadParquet to read rows");
                return Task.CompletedTask;
            },

            [Key(nameof(Ingestion), nameof(Ingestion.LoadCsv))] = _ =>
            {
                using var query = Ingestion.LoadCsv(csvPath);
                using var frame = query.Collect();
                Assert.That(frame.RowCount, Is.GreaterThan(0), "expected LoadCsv to read rows");
                return Task.CompletedTask;
            },

            [Key(nameof(Ingestion), nameof(Ingestion.StreamChunks))] = async ct =>
            {
                int rows = 0;
                await foreach (var chunk in Ingestion.StreamChunks(requestsPath, ChunkSize, ct))
                {
                    using (chunk)
                        rows += chunk.RowCount;
                }
                Assert.That(rows, Is.GreaterThan(0), "expected StreamChunks to yield rows");
            },

            // --- Scenarios ---------------------------------------------------------
            [Key(nameof(Scenarios), nameof(Scenarios.A))] = _ => AssertScenario(Scenarios.A, "A"),
            [Key(nameof(Scenarios), nameof(Scenarios.B))] = _ => AssertScenario(Scenarios.B, "B"),
            [Key(nameof(Scenarios), nameof(Scenarios.C))] = _ => AssertScenario(Scenarios.C, "C"),
            [Key(nameof(Scenarios), nameof(Scenarios.D))] = _ => AssertScenario(Scenarios.D, "D"),

            [Key(nameof(Scenarios), nameof(Scenarios.All))] = _ =>
            {
                Assert.That(Scenarios.All, Has.Count.EqualTo(4));
                Assert.That(Scenarios.All.Select(s => s.Id), Is.EquivalentTo(new[] { "A", "B", "C", "D" }));
                return Task.CompletedTask;
            },

            [Key(nameof(Scenarios), nameof(Scenarios.Get))] = _ =>
            {
                Assert.That(Scenarios.Get("a").Id, Is.EqualTo("A"), "Get should be case-insensitive");
                return Task.CompletedTask;
            },

            // --- Scenario data shapes ----------------------------------------------
            [nameof(ServiceEvent)] = _ =>
            {
                var stamp = new DateTimeOffset(2025, 6, 15, 14, 5, 0, TimeSpan.Zero);
                var serviceEvent = new ServiceEvent
                {
                    Timestamp = stamp,
                    Service = "orders",
                    EventType = "latency_spike",
                    Magnitude = 5.0,
                };

                Assert.Multiple(() =>
                {
                    Assert.That(serviceEvent.Timestamp, Is.EqualTo(stamp));
                    Assert.That(serviceEvent.Service, Is.EqualTo("orders"));
                    Assert.That(serviceEvent.EventType, Is.EqualTo("latency_spike"));
                    Assert.That(serviceEvent.Magnitude, Is.EqualTo(5.0));
                });
                return Task.CompletedTask;
            },

            [nameof(IncidentScenario)] = _ =>
            {
                var scenarioStart = new DateTimeOffset(2025, 6, 15, 14, 10, 0, TimeSpan.Zero);
                var scenarioEnd = scenarioStart.AddMinutes(10);
                var events = new[] { new ServiceEvent { Service = "gateway", EventType = "deploy" } };
                var incidentScenario = new IncidentScenario
                {
                    Id = "Z",
                    Name = "Constructed shape",
                    IncidentStart = scenarioStart,
                    IncidentEnd = scenarioEnd,
                    Events = events,
                    AffectedServices = ["gateway"],
                };

                Assert.Multiple(() =>
                {
                    Assert.That(incidentScenario.Id, Is.EqualTo("Z"));
                    Assert.That(incidentScenario.Name, Is.EqualTo("Constructed shape"));
                    Assert.That(incidentScenario.IncidentStart, Is.EqualTo(scenarioStart));
                    Assert.That(incidentScenario.IncidentEnd, Is.EqualTo(scenarioEnd));
                    Assert.That(incidentScenario.Events, Is.SameAs(events));
                    Assert.That(incidentScenario.AffectedServices, Is.EquivalentTo(new[] { "gateway" }));
                });
                return Task.CompletedTask;
            },

            // --- StreamixScenarios --------------------------------------------------
            [Key(nameof(StreamixScenarios), nameof(StreamixScenarios.RunFaultTolerantStreaming))] = async ct =>
            {
                var summary = await StreamixScenarios.RunFaultTolerantStreaming(tempDir, scenario, ChunkSize, ct);
                Assert.Multiple(() =>
                {
                    Assert.That(summary.TotalRows, Is.GreaterThan(0), "expected the incident window to contain rows");
                    Assert.That(summary.ChunksProcessed, Is.GreaterThan(0));
                });
            },

            [Key(nameof(StreamixScenarios), nameof(StreamixScenarios.RunWindowedAnalytics))] = async ct =>
            {
                var summary = await StreamixScenarios.RunWindowedAnalytics(tempDir, scenario, ChunkSize, ct);
                Assert.That(summary.Windows, Is.Not.Empty, "expected at least one event-time window");
                Assert.That(summary.TotalRows, Is.GreaterThan(0));
            },

            [Key(nameof(StreamixScenarios), nameof(StreamixScenarios.RunOnlineAutoDiffLearning))] = async ct =>
            {
                var summary = await StreamixScenarios.RunOnlineAutoDiffLearning(tempDir, scenario, batchSize: 512, epochs: 1, ct);
                Assert.That(summary.TrainingBatches, Is.GreaterThan(0), "expected at least one training batch");
                Assert.That(float.IsFinite(summary.FinalLoss), Is.True, $"FinalLoss was not finite: {summary.FinalLoss}");
            },
        };

        static Task AssertScenario(IncidentScenario actual, string expectedId)
        {
            Assert.That(actual.Id, Is.EqualTo(expectedId));
            Assert.That(actual.Events, Is.Not.Empty, $"scenario {expectedId} should declare at least one event");
            Assert.That(actual.AffectedServices, Is.Not.Empty, $"scenario {expectedId} should name at least one affected service");
            return Task.CompletedTask;
        }
    }
}