using TypeSafeAI;

namespace NivaraChat.Modes;

public static class TypesafeMode
{
    private const string OllayaUrl = "http://localhost:11435";
    private const string OllayaModel = "laya:en";

    public static async Task Run(ModeContext ctx)
    {
        Console.WriteLine("=== NivaraChat — TypeSafeAI News Assessment ===\n");
        Console.WriteLine($"Connecting to Ollaya at {OllayaUrl} (model: {OllayaModel})...\n");

        var client = new TypeSafeClient(new TypeSafeClientOptions
        {
            ApiKey = "ollaya-local",
            BaseUrl = new Uri(OllayaUrl),
            DefaultModel = OllayaModel,
            Timeout = TimeSpan.FromSeconds(30),
            MaxRetries = 1,
        });

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

        var newsItems = new[]
        {
            "Laya:en v2.1 released — TypeSafe AI ships updated System One decision model with 15% faster inference and improved calibration on edge cases. Available now via Ollaya local server.",
            "New research paper explores whether System One decision models can replace traditional LLM-based classification pipelines in high-throughput production environments, using Jev as a case study.",
            "Benchmark results: Jev-1.13.0 achieves 94.2% accuracy on the NewsIntent benchmark, outperforming GPT-4o-mini by 3.1 points while running 40x faster on CPU hardware.",
            "Opinion piece: 'The rise of decision models like Laya signals a shift from text generation to structured judgment — but we need better tooling for confidence calibration in production.'",
            "TypeSafe AI announces Jev-1.14.0-beta with support for multi-question batch evaluation, enabling developers to assess 100+ news items in a single API call with parallel execution.",
            "Community discussion: comparing Ollaya (local decision model server) vs. cloud-based TypeSafe API for latency-sensitive applications. Key trade-off: data sovereignty vs. model freshness.",
        };

        Console.WriteLine($"Assessing {newsItems.Length} AI/ML news items...\n");
        Console.WriteLine(new string('─', 80));

        for (int i = 0; i < newsItems.Length; i++)
        {
            var item = newsItems[i];
            Console.WriteLine($"\n[{i + 1}] {item}\n");

            try
            {
                var result = await client.SystemOneAsync(item, q);

                var cat = result.Get(category);
                var release = result.Get(isRelease);
                var benchmarks = result.Get(hasBenchmarks);
                var sig = result.Get(significance);
                var rel = result.Get(relevance);
                var act = result.Get(actionable);

                Console.WriteLine($"  Category:      {cat.Choice} (confidence: {cat.Confidence:P0})");
                Console.WriteLine($"  Is Release:    {release.Probability:P0}");
                Console.WriteLine($"  Has Benchmarks:{benchmarks.Probability:P0}");
                Console.WriteLine($"  Significance:  {sig.Score:F1} / 3.0 (confidence: {sig.Confidence:P0})");
                Console.WriteLine($"  Relevance:     {rel.Score:F1} / 2.0 (confidence: {rel.Confidence:P0})");
                Console.WriteLine($"  Actionable:    {act.Probability:P0}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  [ERROR] {ex.GetType().Name}: {ex.Message}");
            }

            Console.WriteLine(new string('─', 80));
        }

        Console.WriteLine("\nDone.");
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
