using System.Diagnostics;
using TypeSafeAI;

namespace NivaraInference;

/// <summary>
/// TypeSafeAI news assessment against a decision-model engine. Ollaya and Ollama 0.35+ expose the
/// same TypeSafe <c>/v1/systemone</c> API, so the same client runs on either — only the base URL and
/// the model name change. Each engine has a default model (Ollaya <c>laya:en</c>, Ollama
/// <c>tev1:0.8b</c>); <c>--model</c> names another decision model on that engine, and <c>--url</c>
/// points it elsewhere (a different port, or a remote TypeSafe endpoint).
/// <para>
/// Engines are a table, not a set of CLI flags: the CLI takes one as a positional
/// (<c>typesafe &lt;engine&gt; [mode]</c>), so adding a third engine is one row here.
/// </para>
/// </summary>
public static class TypeSafe
{
    /// <summary>Where to send the request, and what to call it by default.</summary>
    private sealed record Engine(string Name, string Url, string Model);

    private static readonly Dictionary<string, Engine> Engines = new(StringComparer.OrdinalIgnoreCase)
    {
        ["ollaya"] = new("ollaya", "http://localhost:11435", "laya:en"),
        ["ollama"] = new("ollama", "http://localhost:11434", "tev1:0.8b"),
    };

    /// <summary>Engine used when the CLI is given none.</summary>
    public const string DefaultEngine = "ollaya";

    /// <summary>Whether the CLI's first <c>typesafe</c> positional names an engine or a mode.</summary>
    public static bool IsEngine(string name) => Engines.ContainsKey(name);

    /// <summary>Engine names, for the usage text and the unsupported-mode message.</summary>
    public static IEnumerable<string> EngineNames => Engines.Values.Select(e => e.Name);

    private static readonly string[] NewsItems =
    {
        "Laya:en v2.1 released — TypeSafe AI ships updated System One decision model with 15% faster inference and improved calibration on edge cases. Available now via Ollaya local server.",
        "New research paper explores whether System One decision models can replace traditional LLM-based classification pipelines in high-throughput production environments, using Jev as a case study.",
        "Benchmark results: Jev-1.13.0 achieves 94.2% accuracy on the NewsIntent benchmark, outperforming GPT-4o-mini by 3.1 points while running 40x faster on CPU hardware.",
        "Opinion piece: 'The rise of decision models like Laya signals a shift from text generation to structured judgment — but we need better tooling for confidence calibration in production.'",
        "TypeSafe AI announces Jev-1.14.0-beta with support for multi-question batch evaluation, enabling developers to assess 100+ news items in a single API call with parallel execution.",
        "Community discussion: comparing Ollaya (local decision model server) vs. cloud-based TypeSafe API for latency-sensitive applications. Key trade-off: data sovereignty vs. model freshness.",
    };

    /// <summary>The question set, plus the typed handles the demo reads its answers through.</summary>
    private sealed record NewsQuestions(
        QuestionSet Set,
        ChoiceHandle<NewsCategory> Category,
        NoulHandle IsRelease,
        NoulHandle HasBenchmarks,
        ScoreHandle Significance,
        ScoreHandle Relevance,
        NoulHandle Actionable);

    public static async Task<int> Run(string engine, string? urlOverride, string? modelOverride)
    {
        var (target, url, model) = Resolve(engine, urlOverride, modelOverride);

        Console.WriteLine("=== NivaraInference — TypeSafeAI News Assessment ===\n");
        Console.WriteLine($"Connecting to {DisplayName(target)} at {url} (model: {model})...\n");

        var client = CreateClient(target, url, model);
        var questions = BuildQuestions();
        var q = questions.Set;

        Console.WriteLine($"Assessing {NewsItems.Length} AI/ML news items...\n");
        Console.WriteLine(new string('─', 80));

        int failures = 0;
        for (int i = 0; i < NewsItems.Length; i++)
        {
            var item = NewsItems[i];
            Console.WriteLine($"\n[{i + 1}] {item}\n");

            try
            {
                var result = await client.SystemOneAsync(item, q);

                var cat = result.Get(questions.Category);
                var release = result.Get(questions.IsRelease);
                var benchmarks = result.Get(questions.HasBenchmarks);
                var sig = result.Get(questions.Significance);
                var rel = result.Get(questions.Relevance);
                var act = result.Get(questions.Actionable);

                Console.WriteLine($"  Category:      {cat.Choice} (confidence: {cat.Confidence:P0})");
                Console.WriteLine($"  Is Release:    {release.Probability:P0}");
                Console.WriteLine($"  Has Benchmarks:{benchmarks.Probability:P0}");
                Console.WriteLine($"  Significance:  {sig.Score:F1} / 3.0 (confidence: {sig.Confidence:P0})");
                Console.WriteLine($"  Relevance:     {rel.Score:F1} / 2.0 (confidence: {rel.Confidence:P0})");
                Console.WriteLine($"  Actionable:    {act.Probability:P0}");
            }
            catch (Exception ex)
            {
                failures++;
                Console.WriteLine($"  [ERROR] {ex.GetType().Name}: {ex.Message}");
            }

            Console.WriteLine(new string('─', 80));
        }

        if (failures == NewsItems.Length)
        {
            Console.Error.WriteLine($"\nAll {failures} assessments failed. Is the {DisplayName(target)} engine running at {url}?");
            return 1;
        }

        Console.WriteLine("\nDone.");
        return 0;
    }

    /// <summary>
    /// The engine named on the command line, with the overrides applied. An unknown engine cannot
    /// reach here: the CLI only reads a positional as an engine when <see cref="IsEngine"/> is true,
    /// so anything else is treated as a mode and rejected by the mode allow-list.
    /// </summary>
    private static (Engine Target, string Url, string Model) Resolve(
        string engine, string? urlOverride, string? modelOverride)
    {
        var target = Engines[engine];
        return (target, urlOverride ?? target.Url, modelOverride ?? target.Model);
    }

    private static string DisplayName(Engine engine) =>
        char.ToUpperInvariant(engine.Name[0]) + engine.Name[1..];

    private static TypeSafeClient CreateClient(Engine engine, string url, string model) =>
        new(new TypeSafeClientOptions
        {
            ApiKey = $"{engine.Name}-local",
            BaseUrl = new Uri(url),
            DefaultModel = model,
            Timeout = TimeSpan.FromSeconds(30),
            MaxRetries = 1,
        });

    private static NewsQuestions BuildQuestions()
    {
        var q = new QuestionSet();
        var category = q.Choice<NewsCategory>("What topic does this news item cover?");
        var isRelease = q.Noul("Does this describe a model release or announcement?",
            yes: "Explicitly announces a new model or version",
            no: "Does not announce a release");
        var hasBenchmarks = q.Noul("Does this mention benchmark results or performance numbers?");
        var significance = q.Score("How significant is this development for the AI/ML field?",
            "minor", "moderate", "major", "breakthrough");
        var relevance = q.Score("How relevant is this to production AI/ML engineering?",
            "low", "medium", "high");
        var actionable = q.Noul("Does this contain actionable technical details engineers can use?");
        return new NewsQuestions(q, category, isRelease, hasBenchmarks, significance, relevance, actionable);
    }
}

public enum NewsCategory
{
    [Label("model-release", Description = "New model or version announcement")]
    ModelRelease,
    [Label("research", Description = "Academic or industry research")]
    ResearchPaper,
    [Label("benchmark", Description = "Performance benchmarks and comparisons")]
    Benchmark,
    [Label("opinion", Description = "Opinion pieces and commentary")]
    Opinion,
    [Label("community", Description = "Community discussions and tooling")]
    Other,
}
