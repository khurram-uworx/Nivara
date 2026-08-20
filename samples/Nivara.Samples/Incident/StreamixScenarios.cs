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
/// Streamix integration scenarios for the Nivara Incident Lab.
/// Demonstrates fault-tolerant streaming, event-time windowed analytics,
/// and online AutoDiff learning using the Nivara × Streamix bridge.
/// </summary>
public static class StreamixScenarios
{
    /// <summary>
    /// Scenario 1: Fault-tolerant streaming with retries and checkpointing.
    /// Streams incident data through a Flux pipeline with retry and checkpoint operators.
    /// </summary>
    public static async Task<StreamingSummary> RunFaultTolerantStreaming(
        string datasetPath,
        IncidentScenario scenario,
        int chunkSize = 10000,
        CancellationToken ct = default)
    {
        var incidentStart = scenario.IncidentStart.Ticks;
        var incidentEnd = scenario.IncidentEnd.Ticks;

        int rowCount = 0;
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
            .Checkpoint("incident-stream")
            .ForEachAsync(chunk =>
            {
                Interlocked.Add(ref rowCount, chunk.RowCount);
                Interlocked.Increment(ref chunkCount);
            }, ct);

        return new StreamingSummary(rowCount, chunkCount);
    }

    /// <summary>
    /// Scenario 2: Event-time windowed analytics using ToFluxWithTimestamp + WindowByTime.
    /// Demonstrates timestamped streaming with event-time windowing, Nivara columnar analytics
    /// (RollingMean) inside each window, and per-window aggregation.
    /// </summary>
    public static async Task<WindowedAnalyticsSummary> RunWindowedAnalytics(
        string datasetPath,
        IncidentScenario scenario,
        int chunkSize = 10000,
        CancellationToken ct = default)
    {
        var incidentStart = scenario.IncidentStart.Ticks;
        var incidentEnd = scenario.IncidentEnd.Ticks;

        var windowResults = new List<WindowResult>();
        int totalRows = 0;

        var query = Ingestion.LoadParquet(Path.Combine(datasetPath, "requests.parquet"))
            .Filter(ColumnExpressions.Col("Timestamp") >= ColumnExpressions.Lit(incidentStart))
            .Filter(ColumnExpressions.Col("Timestamp") <= ColumnExpressions.Lit(incidentEnd));

        await query
            .ToFluxWithTimestamp(
                row => new DateTimeOffset(row.GetValue<long>("Timestamp"), TimeSpan.Zero),
                chunkSize: chunkSize)
            .WindowByTime(TimeSpan.FromMinutes(5), TimeSpan.FromMinutes(1))
            .FlatMap(async window =>
            {
                var frame = await window.ToNivaraFrameAsync(ct);
                if (frame.RowCount == 0)
                    return WindowResult.Empty;

                using var result = frame.AsQueryFrame()
                    .RollingMean("DurationMs", "RollingAvgDuration", windowSize: 10, minPeriods: 1)
                    .Collect();

                var durations = result.GetColumn<double>("DurationMs");
                var rollingAvg = result.GetColumn<double>("RollingAvgDuration");
                var statusCodes = result.GetColumn<int>("StatusCode");
                var timestamps = result.GetColumn<long>("Timestamp");

                double avgDuration = durations.Average();
                double lastRollingAvg = rollingAvg[rollingAvg.Length - 1];
                int errorCount = 0;
                for (int i = 0; i < statusCodes.Length; i++)
                    if (statusCodes[i] >= 500) errorCount++;
                double errorRate = (double)errorCount / statusCodes.Length;

                var windowStart = new DateTimeOffset(timestamps[0], TimeSpan.Zero);
                var windowEnd = new DateTimeOffset(timestamps[timestamps.Length - 1], TimeSpan.Zero);

                Interlocked.Add(ref totalRows, result.RowCount);
                return new WindowResult(windowStart, windowEnd, result.RowCount, avgDuration, lastRollingAvg, errorRate);
            })
            .ForEachAsync(result =>
            {
                lock (windowResults)
                {
                    windowResults.Add(result);
                }
            }, ct);

        return new WindowedAnalyticsSummary(totalRows, windowResults);
    }

    /// <summary>
    /// Scenario 3: Online AutoDiff learning over streaming mini-batches.
    /// Demonstrates the bridge between Streamix streaming and Nivara AutoDiff
    /// by training a simple linear model on streamed incident data.
    /// </summary>
    public static async Task<AutoDiffSummary> RunOnlineAutoDiffLearning(
        string datasetPath,
        IncidentScenario scenario,
        int batchSize = 128,
        int epochs = 3,
        CancellationToken ct = default)
    {
        var incidentStart = scenario.IncidentStart.Ticks;
        var incidentEnd = scenario.IncidentEnd.Ticks;

        var model = new Linear<float>(1, 1);
        var optimizer = new Adam<float>((float)1e-3);
        optimizer.AddParameterGroup(model.GetParameters().Values);
        var lossFn = new MSELoss<float>(Reduction.Mean);

        int totalBatches = 0;
        float lastLoss = 0f;

        var query = Ingestion.LoadParquet(Path.Combine(datasetPath, "requests.parquet"))
            .Filter(ColumnExpressions.Col("Timestamp") >= ColumnExpressions.Lit(incidentStart))
            .Filter(ColumnExpressions.Col("Timestamp") <= ColumnExpressions.Lit(incidentEnd));

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            if (ct.IsCancellationRequested) break;

            await query
                .ToFluxRows(chunkSize: 10000)
                .BufferFrames(batchSize)
                .Map(frame =>
                {
                    var durations = frame.GetColumn<double>("DurationMs").ToArray();
                    var statusCodes = frame.GetColumn<int>("StatusCode").ToArray();

                    var inputs = new float[durations.Length];
                    var targets = new float[statusCodes.Length];
                    for (int i = 0; i < durations.Length; i++)
                    {
                        inputs[i] = (float)(durations[i] / 1000.0);
                        targets[i] = statusCodes[i] >= 500 ? 1f : 0f;
                    }

                    using (GradientUtils.Grad())
                    {
                        var inputTensor = ReverseGradTensor<float>.FromMatrix(inputs, inputs.Length, 1);
                        var pred = model.Forward(inputTensor);
                        var targetTensor = ReverseGradTensor<float>.FromMatrix(targets, targets.Length, 1);
                        var loss = lossFn.Forward(pred, targetTensor);
                        loss.Backward();
                        optimizer.Step();
                        optimizer.ZeroGrad();
                        Interlocked.Increment(ref totalBatches);
                        return loss[0];
                    }
                })
                .ForEachAsync(lossValue =>
                {
                    lastLoss = lossValue;
                }, ct);
        }

        return new AutoDiffSummary(totalBatches, lastLoss, model);
    }

    public sealed record StreamingSummary(int TotalRows, int ChunksProcessed);

    public sealed record WindowResult(
        DateTimeOffset WindowStart,
        DateTimeOffset WindowEnd,
        int RowCount,
        double AverageDurationMs,
        double RollingAvgDurationMs,
        double ErrorRate)
    {
        public static readonly WindowResult Empty = new(default, default, 0, 0, 0, 0);
    }

    public sealed record WindowedAnalyticsSummary(int TotalRows, IReadOnlyList<WindowResult> Windows);

    public sealed record AutoDiffSummary(int TrainingBatches, float FinalLoss, Linear<float> Model);
}
