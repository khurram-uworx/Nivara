using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using Nivara.Expressions;
using Nivara.Query;
using Nivara.Streamix;
using Streamix;

namespace Nivara.Samples.Incident;

/// <summary>
/// Three streaming scenarios demonstrating the Nivara × Streamix bridge:
/// 1. Fault-tolerant streaming with retries and checkpointing
/// 2. Event-time timestamped streaming with per-chunk analytics
/// 3. Online AutoDiff learning over streaming mini-batches
/// </summary>
public static class StreamixScenarios
{
    /// <summary>
    /// Scenario 1: Fault-tolerant streaming with retries and checkpointing.
    /// Wraps a Parquet source in a Flux with retry logic and checkpoint logging,
    /// then processes each chunk via ForEachAsync.
    /// </summary>
    public static async Task<int> RunFaultTolerantStreaming(
        string datasetPath,
        IncidentScenario scenario,
        int chunkSize = 10_000,
        CancellationToken ct = default)
    {
        var incidentStart = scenario.IncidentStart.Ticks;
        var incidentEnd = scenario.IncidentEnd.Ticks;

        int totalRows = 0;
        int chunkCount = 0;

        var query = Ingestion.LoadParquet(Path.Combine(datasetPath, "requests.parquet"))
            .Filter(ColumnExpressions.Col("Timestamp") >= ColumnExpressions.Lit(incidentStart))
            .Filter(ColumnExpressions.Col("Timestamp") <= ColumnExpressions.Lit(incidentEnd));

        await query.ToFlux(chunkSize: chunkSize)
            .Retry(3, (attempt, ex) =>
            {
                Console.WriteLine($"  [retry] attempt {attempt}: {ex.Message}");
                return TimeSpan.FromMilliseconds(100 * attempt);
            })
            .Checkpoint("incident-chunk")
            .ForEachAsync(chunk =>
            {
                Interlocked.Add(ref totalRows, chunk.RowCount);
                Interlocked.Increment(ref chunkCount);
            }, ct);

        return totalRows;
    }

    /// <summary>
    /// Scenario 2: Event-time timestamped streaming.
    /// Attaches a DateTimeOffset to each row, then computes per-chunk
    /// statistics (min/max timestamp, average DurationMs) via Map.
    /// </summary>
    public static async Task<List<ChunkStats>> RunTimestampedAnalytics(
        string datasetPath,
        IncidentScenario scenario,
        int chunkSize = 10_000,
        CancellationToken ct = default)
    {
        var incidentStart = scenario.IncidentStart.Ticks;
        var incidentEnd = scenario.IncidentEnd.Ticks;
        var results = new System.Collections.Concurrent.ConcurrentBag<ChunkStats>();

        var query = Ingestion.LoadParquet(Path.Combine(datasetPath, "requests.parquet"))
            .Filter(ColumnExpressions.Col("Timestamp") >= ColumnExpressions.Lit(incidentStart))
            .Filter(ColumnExpressions.Col("Timestamp") <= ColumnExpressions.Lit(incidentEnd));

        await query
            .ToFluxWithTimestamp(
                row => new DateTimeOffset(row.GetValue<long>("Timestamp"), TimeSpan.Zero),
                chunkSize: chunkSize)
            .Map(timestamped =>
            {
                var row = timestamped.Value;
                return new ChunkStats
                {
                    Timestamp = timestamped.Timestamp,
                    DurationMs = row.GetValue<double>("DurationMs"),
                    StatusCode = row.GetValue<int>("StatusCode"),
                    Service = row.GetValue<string>("Service") ?? ""
                };
            })
            .ForEachAsync(stats =>
            {
                results.Add(stats);
            }, ct);

        return results.ToList();
    }

    /// <summary>
    /// Scenario 3: Online AutoDiff learning over streaming mini-batches.
    /// Trains a simple Linear model to predict error probability (StatusCode >= 500)
    /// from normalized DurationMs, processing one mini-batch at a time through a Flux stream.
    /// </summary>
    public static async Task<int> RunOnlineAutoDiffLearning(
        string datasetPath,
        IncidentScenario scenario,
        int batchSize = 128,
        int epochs = 3,
        CancellationToken ct = default)
    {
        var incidentStart = scenario.IncidentStart.Ticks;
        var incidentEnd = scenario.IncidentEnd.Ticks;

        var query = Ingestion.LoadParquet(Path.Combine(datasetPath, "requests.parquet"))
            .Filter(ColumnExpressions.Col("Timestamp") >= ColumnExpressions.Lit(incidentStart))
            .Filter(ColumnExpressions.Col("Timestamp") <= ColumnExpressions.Lit(incidentEnd));

        var model = new Linear<float>(1, 1);
        var optimizer = new Adam<float>((float)1e-3);
        optimizer.AddParameterGroup(model.GetParameters().Values, (float)1e-3);
        var lossFn = new MSELoss<float>(Reduction.Mean);

        int totalBatches = 0;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            await query
                .ToFluxRows(chunkSize: 10_000)
                .BufferFrames(batchSize)
                .Map(async frame =>
                {
                    var durations = frame.GetColumn<double>("DurationMs").ToArray();
                    var statusCodes = frame.GetColumn<int>("StatusCode").ToArray();

                    double maxDur = 0;
                    foreach (var d in durations)
                        if (d > maxDur) maxDur = d;
                    if (maxDur == 0) maxDur = 1.0;

                    var inputs = new float[durations.Length];
                    var targets = new float[statusCodes.Length];
                    for (int i = 0; i < durations.Length; i++)
                    {
                        inputs[i] = (float)(durations[i] / maxDur);
                        targets[i] = statusCodes[i] >= 500 ? 1.0f : 0.0f;
                    }

                    ReverseGradTensor<float> loss;
                    using (GradientUtils.Grad())
                    {
                        var inputTensor = ReverseGradTensor<float>.FromArray(inputs);
                        var pred = model.Forward(inputTensor);
                        var targetTensor = ReverseGradTensor<float>.FromArray(targets, requiresGrad: false);
                        loss = lossFn.Forward(pred, targetTensor);
                    }

                    loss.Backward();
                    optimizer.Step();
                    optimizer.ZeroGrad();

                    Interlocked.Increment(ref totalBatches);
                    return loss[0];
                })
                .ForEachAsync(_ => { }, ct);
        }

        return totalBatches;
    }

    public sealed class ChunkStats
    {
        public DateTimeOffset Timestamp { get; init; }
        public double DurationMs { get; init; }
        public int StatusCode { get; init; }
        public string Service { get; init; } = "";
    }
}
