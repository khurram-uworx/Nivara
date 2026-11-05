using NivaraChat.Helpers;
using System.Text;

namespace NivaraChat.Modes;

/// <summary>
/// --rag-multiview: #914. Embeds four views of every chunk (Content, Title, Section, Code), scores
/// them independently, fuses the scores late, names the view that carried the hit, and reports where
/// the views disagree. No LLM: the point is the retrieval signal, not the answer.
/// </summary>
public static class RagMultiViewMode
{
    public static async Task Run(ModeContext ctx)
    {
        Console.WriteLine("=== NivaraChat — Multi-View Retrieval (#914) ===\n");

        if (!FusionWeights.TryParse(ctx.Weights, out var weights, out var error))
        {
            Console.WriteLine($"Invalid --weights: {error}");
            Console.WriteLine("Expected a list like content=0.5,title=0.2,section=0.2,code=0.1.");
            return;
        }

        using var index = await MultiViewIndex.BuildAsync(ctx);
        if (index is null) return;

        Console.WriteLine($"  Files:    {index.FileCount}");
        Console.WriteLine($"  Chunks:   {index.Chunks.Count}");
        Console.WriteLine($"  Views:    content {index.ViewCount(ViewSpace.Content)}, title {index.ViewCount(ViewSpace.Title)}, "
            + $"section {index.ViewCount(ViewSpace.Section)}, code {index.ViewCount(ViewSpace.Code)}");
        Console.WriteLine($"  Weights:  content {weights.Content:0.00}, title {weights.Title:0.00}, "
            + $"section {weights.Section:0.00}, code {weights.Code:0.00} (renormalized per chunk over the views it has)");
        Console.WriteLine($"  Bands:    rank Near <= {MultiViewScoring.NearBandBound:0.00}, Distinct <= {MultiViewScoring.DistinctBandBound:0.00}; "
            + $"agreement Agree <= {MultiViewExplorer.AgreeBound:0.00}, Mild <= {MultiViewExplorer.MildBound:0.00}");
        Console.WriteLine();

        if (!index.HasVisibleChunks)
        {
            Console.WriteLine($"No chunks visible under ACL group '{ctx.AclGroup}'.");
            return;
        }

        var reportPath = ctx.ReportPath is null ? null : ResolveReportPath(ctx.ReportPath);
        var report = new StringBuilder();
        report.AppendLine("# Multi-view retrieval report");
        report.AppendLine();
        report.AppendLine($"- Corpus: {index.FileCount} files, {index.Chunks.Count} chunks");
        report.AppendLine($"- Views: content {index.ViewCount(ViewSpace.Content)}, title {index.ViewCount(ViewSpace.Title)}, "
            + $"section {index.ViewCount(ViewSpace.Section)}, code {index.ViewCount(ViewSpace.Code)}");
        report.AppendLine($"- Weights: content {weights.Content:0.00}, title {weights.Title:0.00}, "
            + $"section {weights.Section:0.00}, code {weights.Code:0.00}, renormalized per chunk over the views it has");
        report.AppendLine($"- Bands: rank Near <= {MultiViewScoring.NearBandBound:0.00}, Distinct <= {MultiViewScoring.DistinctBandBound:0.00}; "
            + $"agreement Agree <= {MultiViewExplorer.AgreeBound:0.00}, Mild <= {MultiViewExplorer.MildBound:0.00}");
        report.AppendLine();

        void RunQuery(string query)
        {
            var (ranked, elapsedMs) = index.Retrieve(query, weights, ctx.Explore ? int.MaxValue : ctx.TopK);
            var top = ranked.Take(ctx.TopK).ToArray();
            PrintResults(top, elapsedMs);

            IReadOnlyList<AgreementBucket>? buckets = null;
            if (ctx.Explore)
            {
                buckets = MultiViewExplorer.Bucketize(ranked);
                Console.WriteLine($"  Agreement over all {ranked.Count} visible chunks:");
                Console.WriteLine(Indent(MultiViewExplorer.Describe(buckets), "  "));
                Console.WriteLine();
            }

            if (reportPath is not null)
            {
                report.Append(BuildSection(query, top, buckets));
                WriteReport(reportPath, report);
                Console.WriteLine($"  Report written to {reportPath}\n");
            }
        }

        Console.WriteLine($"Top-K: {ctx.TopK} chunks per query.{(ctx.Explore ? " Agreement over the whole visible corpus." : string.Empty)}");
        Console.WriteLine("Type a question (or 'quit' to exit):\n");

        if (ctx.SingleShotText != null)
        {
            RunQuery(ctx.SingleShotText);
            return;
        }

        while (true)
        {
            Console.Write("> ");
            var query = Console.ReadLine()?.Trim();
            if (string.IsNullOrEmpty(query) || query == "quit") break;
            RunQuery(query);
        }
    }

    static void PrintResults(IReadOnlyList<MultiViewHit> hits, long elapsedMs)
    {
        Console.WriteLine($"  Retrieval: {elapsedMs} ms - {hits.Count} hit(s)\n");
        if (hits.Count == 0)
        {
            Console.WriteLine("  No chunk scored for this query.\n");
            return;
        }

        Console.WriteLine("   #  fused   band      best     content  title    section  code     maxdiv");
        Console.WriteLine("  --  ------  --------  -------  -------  -------  -------  -------  ------");
        var rank = 0;
        foreach (var hit in hits)
        {
            rank++;
            Console.WriteLine(
                $"  {rank,2}  {Fmt(hit.Fused)}  {hit.RankBand,-8}  {Best(hit),-7}  "
                + $"{Fmt(hit.Scores.Content)}  {Fmt(hit.Scores.Title)}  {Fmt(hit.Scores.Section)}  "
                + $"{Fmt(hit.Scores.Code)}  {Fmt(hit.MaxDivergence)}");
            Console.WriteLine($"      {hit.Chunk.ChunkId}  -  {hit.Chunk.Section}");
            Console.WriteLine($"      {Preview(hit.Chunk.Content)}");
            if (hit.Chunk.Code is { } code)
                Console.WriteLine($"      code: {Preview(code)}");
            Console.WriteLine();
        }

        Console.WriteLine($"  {MultiViewExplorer.Explain(hits[0])}\n");
    }

    static string BuildSection(string query, IReadOnlyList<MultiViewHit> hits, IReadOnlyList<AgreementBucket>? buckets)
    {
        var builder = new StringBuilder();
        builder.AppendLine($"## Query: {Escape(query)}");
        builder.AppendLine();
        if (hits.Count == 0)
        {
            builder.AppendLine("_No chunk scored for this query._");
            builder.AppendLine();
            return builder.ToString();
        }

        builder.AppendLine("| # | Fused | Band | Best view | Content | Title | Section | Code | Max divergence | Chunk |");
        builder.AppendLine("| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |");
        var rank = 0;
        foreach (var hit in hits)
        {
            rank++;
            builder.AppendLine(
                $"| {rank} | {Fmt(hit.Fused)} | {hit.RankBand} | {Best(hit)} | {Fmt(hit.Scores.Content)} | "
                + $"{Fmt(hit.Scores.Title)} | {Fmt(hit.Scores.Section)} | {Fmt(hit.Scores.Code)} | "
                + $"{Fmt(hit.MaxDivergence)} | {Escape(hit.Chunk.ChunkId)} |");
        }

        builder.AppendLine();
        builder.AppendLine("### Agreement");
        builder.AppendLine();
        if (buckets is null)
        {
            builder.AppendLine("_Run with `--explore` to include disagreement buckets over the whole visible corpus._");
        }
        else
        {
            foreach (var bucket in buckets)
            {
                var bound = bucket.UpperBound is { } upper ? $" (max divergence <= {upper:0.00})" : string.Empty;
                builder.AppendLine($"- {bucket.Name}{bound}: {bucket.Count}");
            }
        }

        builder.AppendLine();
        builder.AppendLine("### Why the top hit won");
        builder.AppendLine();
        builder.AppendLine(MultiViewExplorer.Explain(hits[0]));
        builder.AppendLine();
        return builder.ToString();
    }

    static void WriteReport(string path, StringBuilder content)
    {
        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
        File.WriteAllText(path, content.ToString());
    }

    static string ResolveReportPath(string path)
    {
        if (Path.IsPathRooted(path))
            return path;

        var directory = Path.Combine(ModeHelpers.GetRepoRoot(), "samples", "data", "nivarachat");
        return Path.Combine(directory, path);
    }

    static string Fmt(float? value) => value is { } present ? present.ToString("0.000") : "-";

    static string Best(MultiViewHit hit) => hit.BestSpace?.ToString() ?? "-";

    static string Preview(string text)
    {
        var collapsed = string.Join(' ', text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
        return collapsed.Length <= 120 ? collapsed : collapsed[..117] + "...";
    }

    static string Escape(string text) => text.Replace("|", "\\|").Replace('\n', ' ').Replace('\r', ' ');

    static string Indent(string text, string prefix) =>
        string.Join(Environment.NewLine, text.Split('\n').Select(line => prefix + line.TrimEnd('\r')));
}
