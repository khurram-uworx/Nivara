using Nivara.Silk;

namespace NivaraChat.Helpers;

internal sealed class SilkRetrievalTrainer : IDisposable
{
    internal enum GateMode { None, Stats, Embed }

    readonly SilkRuntime runtime;
    readonly GateMode gate;
    readonly int features;
    readonly float learningRate;
    readonly float weightDecay;
    readonly int maxRows;

    nint Sbuf, Pbuf, Fbuf, Dbuf, Gbuf, Wbuf;

    // Global path.
    readonly float[] theta;
    readonly float[] thetaBest;
    nint thetaBuf, thetaGradBuf, thetaM, thetaV;
    readonly Random rng;

    // Gate path.
    readonly float[] gateWeight;
    readonly float[] gateWeightBest;
    readonly float[] gateBias;
    readonly float[] gateBiasBest;
    nint gateWBuf, gateGradWBuf, gateWm, gateWv;
    nint gateBBuf, gateGradBBuf, gateBm, gateBv;

    int adamStep;
    readonly bool hasGate;

    SilkRetrievalTrainer(GateMode gate, int features, float learningRate, float weightDecay, int maxRows)
    {
        this.gate = gate;
        this.features = features;
        this.learningRate = learningRate;
        this.weightDecay = weightDecay;
        this.maxRows = Math.Max(1, maxRows);
        hasGate = gate != GateMode.None;
        rng = new Random(42);

        runtime = new SilkRuntime();
        Sbuf = runtime.Allocate(maxRows * TrainingSetBuilder.ViewCount);
        Pbuf = runtime.Allocate(maxRows * TrainingSetBuilder.ViewCount);
        Wbuf = runtime.Allocate(TrainingSetBuilder.ViewCount);
        Fbuf = runtime.Allocate(maxRows);
        Dbuf = runtime.Allocate(maxRows);
        Gbuf = runtime.Allocate(maxRows);

        theta = MaskedFusion.LogWeights(FusionWeights.Default);
        thetaBest = new float[theta.Length];
        Array.Copy(theta, thetaBest, theta.Length);
        thetaBuf = runtime.Allocate(theta.Length);
        thetaGradBuf = runtime.Allocate(theta.Length);
        thetaM = runtime.Allocate(theta.Length);
        thetaV = runtime.Allocate(theta.Length);
        runtime.Upload(thetaBuf, theta.AsSpan());
        ZeroDeviceBuffer(thetaM, theta.Length);
        ZeroDeviceBuffer(thetaV, theta.Length);

        gateWeight = hasGate ? new float[TrainingSetBuilder.ViewCount * features] : Array.Empty<float>();
        gateWeightBest = hasGate ? new float[gateWeight.Length] : Array.Empty<float>();
        gateBias = hasGate ? MaskedFusion.LogWeights(FusionWeights.Default) : Array.Empty<float>();
        gateBiasBest = hasGate ? new float[gateBias.Length] : Array.Empty<float>();
        if (hasGate)
        {
            Array.Copy(gateBias, gateBiasBest, gateBias.Length);
            gateWBuf = runtime.Allocate(gateWeight.Length);
            gateGradWBuf = runtime.Allocate(gateWeight.Length);
            gateWm = runtime.Allocate(gateWeight.Length);
            gateWv = runtime.Allocate(gateWeight.Length);
            gateBBuf = runtime.Allocate(gateBias.Length);
            gateGradBBuf = runtime.Allocate(gateBias.Length);
            gateBm = runtime.Allocate(gateBias.Length);
            gateBv = runtime.Allocate(gateBias.Length);
            runtime.Upload(gateWBuf, gateWeight.AsSpan());
            runtime.Upload(gateBBuf, gateBias.AsSpan());
            ZeroDeviceBuffer(gateWm, gateWeight.Length);
            ZeroDeviceBuffer(gateWv, gateWeight.Length);
            ZeroDeviceBuffer(gateBm, gateBias.Length);
            ZeroDeviceBuffer(gateBv, gateBias.Length);
        }
        else
        {
            gateWBuf = 0; gateGradWBuf = 0; gateWm = 0; gateWv = 0;
            gateBBuf = 0; gateGradBBuf = 0; gateBm = 0; gateBv = 0;
        }
    }

    internal static float[] GateFeatures(GateMode gate, int features, TrainingQuery query)
    {
        if (gate == GateMode.Embed) return query.QueryVector.Length == features
            ? query.QueryVector
            : throw new ArgumentException("Query vector dimension mismatch.");

        var means = new float[TrainingSetBuilder.ViewCount];
        var counts = new int[TrainingSetBuilder.ViewCount];
        var maxima = new float[TrainingSetBuilder.ViewCount];
        Array.Fill(maxima, float.NegativeInfinity);
        for (var row = 0; row < query.RowCount; row++)
        {
            for (var view = 0; view < TrainingSetBuilder.ViewCount; view++)
            {
                var index = row * TrainingSetBuilder.ViewCount + view;
                if (!query.Present[index]) continue;
                var score = query.Scores[index];
                means[view] += score; counts[view]++;
                maxima[view] = MathF.Max(maxima[view], score);
            }
        }
        var vector = new float[TrainingSetBuilder.ViewCount * 2];
        for (var view = 0; view < TrainingSetBuilder.ViewCount; view++)
        {
            vector[view] = counts[view] == 0 ? 0f : means[view] / counts[view];
            vector[TrainingSetBuilder.ViewCount + view] = counts[view] == 0 ? 0f : maxima[view];
        }
        return vector;
    }

    void ZeroDeviceBuffer(nint b, int count)
    {
        if (b == 0) return;
        var zeros = new float[count];
        runtime.Upload(b, zeros.AsSpan());
    }

    public static SilkRetrievalTrainer CreateGlobal(float learningRate, float weightDecay, int maxRows) =>
        new(GateMode.None, 0, learningRate, weightDecay, maxRows);

    public static SilkRetrievalTrainer CreateForStats(float learningRate, float weightDecay, int maxRows) =>
        new(GateMode.Stats, TrainingSetBuilder.ViewCount * 2, learningRate, weightDecay, maxRows);

    public static SilkRetrievalTrainer CreateForEmbed(float learningRate, float weightDecay, int maxRows, int queryDim) =>
        new(GateMode.Embed, queryDim, learningRate, weightDecay, maxRows);

    public string DeviceName => runtime.DeviceName;

    public bool IsGate => gate != GateMode.None;

    public int FeatureCount => gate == GateMode.None ? 0 : features;

    public float[] FeatureVector(TrainingQuery query) => GateFeatures(gate, features, query);

    public float[] Weights
    {
        get
        {
            if (!hasGate)
            {
                var fresh = runtime.Download(thetaBuf, theta.Length);
                return MaskedFusion.Weights(fresh);
            }
            var bias = runtime.Download(gateBBuf, gateBias.Length);
            return MaskedFusion.Weights(bias);
        }
    }

    public float[] WeightsFor(TrainingQuery query)
    {
        if (!hasGate) return Weights;
        var x = GateFeatures(gate, features, query);
        runtime.Download(gateWBuf, gateWeight.Length).CopyTo(gateWeight, 0);
        runtime.Download(gateBBuf, gateBias.Length).CopyTo(gateBias, 0);
        var logits = new float[TrainingSetBuilder.ViewCount];
        for (var k = 0; k < TrainingSetBuilder.ViewCount; k++)
        {
            var acc = gateBias[k];
            for (var j = 0; j < features; j++) acc += gateWeight[k * features + j] * x[j];
            logits[k] = acc;
        }
        return MaskedFusion.Weights(logits);
    }

    public float[] Scores(TrainingQuery query)
    {
        var c = query.RowCount;
        runtime.Upload(Sbuf, query.Scores.AsSpan());
        runtime.Upload(Pbuf, FloatMask(query.Present).AsSpan());

        float[] w;
        if (!hasGate)
        {
            runtime.Download(thetaBuf, theta.Length).CopyTo(theta, 0);
            w = MaskedFusion.Weights(theta.AsSpan());
            runtime.Upload(Wbuf, w.AsSpan());
        }
        else
        {
            runtime.Download(gateWBuf, gateWeight.Length).CopyTo(gateWeight, 0);
            runtime.Download(gateBBuf, gateBias.Length).CopyTo(gateBias, 0);
            var x = GateFeatures(gate, features, query);
            var logits = new float[TrainingSetBuilder.ViewCount];
            for (var k = 0; k < TrainingSetBuilder.ViewCount; k++)
            {
                var acc = gateBias[k];
                for (var j = 0; j < features; j++) acc += gateWeight[k * features + j] * x[j];
                logits[k] = acc;
            }
            w = MaskedFusion.Weights(logits.AsSpan());
            runtime.Upload(Wbuf, w.AsSpan());
        }
        runtime.Upload(Wbuf, w.AsSpan()); // scratch: fusion_forward reads W from Wbuf

        runtime.SetBuffer("fusion_forward", 0, Sbuf);
        runtime.SetBuffer("fusion_forward", 1, Pbuf);
        runtime.SetBuffer("fusion_forward", 2, Wbuf);
        runtime.SetBuffer("fusion_forward", 3, Fbuf);
        runtime.SetBuffer("fusion_forward", 4, Dbuf);
        runtime.SetUInt32("fusion_forward", 5, (uint)c);
        nuint[] global = [(nuint)RoundUp64(c)];
        runtime.Run("fusion_forward", new nuint[] { (nuint)RoundUp64(c) }, new nuint[] { 64 });
        runtime.Finish();

        var f = runtime.Download(Fbuf, c);
        var d = runtime.Download(Dbuf, c);
        for (var i = 0; i < c; i++)
            if (d[i] == 0f)
                throw new DivideByZeroException($"Fusion denominator zero at row {i}.");
        return f;
    }

    static float[] FloatMask(bool[] present) => present.Select(x => x ? 1f : 0f).ToArray();

    public TrainingReport Train(IReadOnlyList<TrainingQuery> train, IReadOnlyList<TrainingQuery> validation, TrainingOptions? options, int topK)
    {
        options ??= new TrainingOptions();
        var usable = Usable(train);
        if (usable.Length == 0)
            return new TrainingReport([], 0, double.NaN, double.NaN, 0, 0, false);

        var shuffled = usable.ToList();
        var history = new List<EpochReport>(options.Epochs);
        var bestScore = double.NegativeInfinity;
        var bestTrainLoss = double.NaN;
        var bestEpoch = 0;
        float[]? bestTheta = null;
        float[]? bestGateW = null;
        float[]? bestGateB = null;
        var sinceBest = 0;
        var random = new Random(options.Seed);
        int steps = 0;

        for (var epoch = 1; epoch <= options.Epochs; epoch++)
        {
            shuffled = shuffled.OrderBy(_ => rng.Next()).ToList();
            var lossSum = 0d;
            var lossCount = 0;
            foreach (var q in shuffled)
            {
                var f = ForwardAndGradient(q, options.LearningRate, options.WeightDecay, out var trainLoss);
                lossSum += trainLoss;
                lossCount++;
                steps++;
            }
            var trainLossAvg = lossCount == 0 ? double.NaN : lossSum / lossCount;
            var valNdcg = validation.Count == 0 ? double.NaN : MeanNdcg(validation, topK);
            history.Add(new EpochReport(epoch, trainLossAvg, valNdcg));

            if (validation.Count == 0)
            {
                bestEpoch = epoch; bestTrainLoss = trainLossAvg; bestScore = valNdcg;
                SnapshotBest(ref bestTheta, ref bestGateW, ref bestGateB);
                if (epoch == options.Epochs) break;
                continue;
            }

            if (valNdcg > bestScore)
            {
                bestScore = valNdcg; bestEpoch = epoch; bestTrainLoss = trainLossAvg;
                sinceBest = 0;
                SnapshotBest(ref bestTheta, ref bestGateW, ref bestGateB);
            }
            else if (++sinceBest >= options.Patience)
            {
                break;
            }
        }

        RestoreBest(bestTheta, bestGateW, bestGateB);
        return new TrainingReport(
            history,
            bestEpoch,
            validation.Count == 0 ? double.NaN : bestScore,
            bestTrainLoss,
            steps,
            usable.Length,
            history.Count < options.Epochs);
    }

    void SnapshotBest(ref float[]? bestTheta, ref float[]? bestGateW, ref float[]? bestGateB)
    {
        if (!hasGate)
        {
            var cur = runtime.Download(thetaBuf, theta.Length);
            bestTheta = new float[cur.Length];
            cur.CopyTo(bestTheta, 0);
        }
        else
        {
            var cw = runtime.Download(gateWBuf, gateWeight.Length);
            var cb = runtime.Download(gateBBuf, gateBias.Length);
            bestGateW = new float[cw.Length]; bestGateB = new float[cb.Length];
            cw.CopyTo(bestGateW, 0); cb.CopyTo(bestGateB, 0);
        }
    }

    void RestoreBest(float[]? bestTheta, float[]? bestGateW, float[]? bestGateB)
    {
        if (bestTheta is not null)
        {
            Array.Copy(bestTheta, theta, theta.Length);
            runtime.Upload(thetaBuf, theta.AsSpan());
        }
        if (bestGateW is not null && bestGateB is not null && hasGate)
        {
            Array.Copy(bestGateW, gateWeight, gateWeight.Length);
            Array.Copy(bestGateB, gateBias, gateBias.Length);
            runtime.Upload(gateWBuf, gateWeight.AsSpan());
            runtime.Upload(gateBBuf, gateBias.AsSpan());
        }
    }

    double ForwardAndGradient(TrainingQuery query, float lr, float weightDecay, out double trainLoss)
    {
        var c = query.RowCount;
        runtime.Upload(Sbuf, query.Scores.AsSpan());
        runtime.Upload(Pbuf, FloatMask(query.Present).AsSpan());

        // World the weights from device.
        runtime.Download(thetaBuf, theta.Length).CopyTo(theta, 0);
        if (!hasGate)
        {
            var gradTheta = new[] { 0f, 0f, 0f, 0f };
            ForwardAndGradientGlobal(query, c, gradTheta, out trainLoss);
            ApplyGlobalUpdate(lr, weightDecay, gradTheta, trainLoss);
        }
        else
        {
            runtime.Download(gateWBuf, gateWeight.Length).CopyTo(gateWeight, 0);
            runtime.Download(gateBBuf, gateBias.Length).CopyTo(gateBias, 0);
            var x = GateFeatures(gate, features, query);
            var gradW = new float[gateWeight.Length];
            var gradB = new float[gateBias.Length];
            ForwardAndGradientGated(query, c, x, gradW, gradB, out trainLoss);
            ApplyGateUpdate(lr, weightDecay, gradW, gradB, trainLoss);
        }
        return trainLoss;
    }

    void ForwardAndGradientGlobal(TrainingQuery query, int c, float[] gradTheta, out double trainLoss)
    {
        var w = MaskedFusion.Weights(theta.AsSpan());
        runtime.Upload(Wbuf, w.AsSpan());
        runtime.SetBuffer("fusion_forward", 0, Sbuf);
        runtime.SetBuffer("fusion_forward", 1, Pbuf);
        runtime.SetBuffer("fusion_forward", 2, Wbuf);
        runtime.SetBuffer("fusion_forward", 3, Fbuf);
        runtime.SetBuffer("fusion_forward", 4, Dbuf);
        runtime.SetUInt32("fusion_forward", 5, (uint)c);
        runtime.Run("fusion_forward", new nuint[] { (nuint)RoundUp64(c) }, new nuint[] { 64 });
        runtime.Finish();
        var f = runtime.Download(Fbuf, c);
        var d = runtime.Download(Dbuf, c);

        var target = query.LossTarget;
        var g = new float[c];
        var softmaxF = MaskedFusion.Weights(f.AsSpan());
        for (var i = 0; i < c; i++) g[i] = softmaxF[i] - target[i];
        trainLoss = ListwiseLoss(f, target);

        runtime.Upload(Gbuf, g.AsSpan());
        runtime.SetBuffer("fusion_wgrad", 0, Sbuf);
        runtime.SetBuffer("fusion_wgrad", 1, Pbuf);
        runtime.SetBuffer("fusion_wgrad", 2, Gbuf);
        runtime.SetBuffer("fusion_wgrad", 3, Fbuf);
        runtime.SetBuffer("fusion_wgrad", 4, Dbuf);
        runtime.SetBuffer("fusion_wgrad", 5, thetaGradBuf);
        runtime.SetUInt32("fusion_wgrad", 6, (uint)c);
        nuint[] global = [4];
        runtime.Run("fusion_wgrad", new nuint[] { 4 }, new nuint[] { 1 });

        var dW = runtime.Download(thetaGradBuf, TrainingSetBuilder.ViewCount);
        for (var i = 0; i < dW.Length; i++)
        {
            var contribution = dW[i];
            var wv = w[i];
            // dL/dtheta[i] = w_i * (contrib_i - sum_v w_v contrib_v)
        }
        var dot = 0f;
        for (var i = 0; i < dW.Length; i++) dot += w[i] * dW[i];
        for (var i = 0; i < gradTheta.Length; i++) gradTheta[i] = w[i] * (dW[i] - dot);
    }

    void ForwardAndGradientGated(TrainingQuery query, int c, float[] x, float[] gradW, float[] gradB, out double trainLoss)
    {
        var thetaHost = gateBias; // placeholder shell; still need gate weight transform
        // Use the gate's raw bias as the fusion weight initializer for the scalar path.
        // Gate stays on CPU/host; forward then runs on the device buffer for symmetry.
        var logits = new float[TrainingSetBuilder.ViewCount];
        runtime.Download(gateWBuf, gateWeight.Length).CopyTo(gateWeight, 0);
        runtime.Download(gateBBuf, gateBias.Length).CopyTo(gateBias, 0);
        for (var k = 0; k < TrainingSetBuilder.ViewCount; k++)
        {
            var acc = gateBias[k];
            for (var j = 0; j < features; j++) acc += gateWeight[k * features + j] * x[j];
            logits[k] = acc;
        }
        var w = MaskedFusion.Weights(logits.AsSpan());
        // Copy into theta scratch so fusion_forward has W to read.
        runtime.Upload(thetaBuf, w.AsSpan()); // use theta buffer as scratch

        runtime.Upload(Sbuf, query.Scores.AsSpan());
        runtime.Upload(Pbuf, FloatMask(query.Present).AsSpan());
        runtime.SetBuffer("fusion_forward", 0, Sbuf);
        runtime.SetBuffer("fusion_forward", 1, Pbuf);
        runtime.SetBuffer("fusion_forward", 2, Wbuf);
        runtime.SetBuffer("fusion_forward", 3, Fbuf);
        runtime.SetBuffer("fusion_forward", 4, Dbuf);
        runtime.SetUInt32("fusion_forward", 5, (uint)c);
        runtime.Run("fusion_forward", new nuint[] { (nuint)RoundUp64(c) }, new nuint[] { 64 });
        runtime.Finish();
        var f = runtime.Download(Fbuf, c);
        var d = runtime.Download(Dbuf, c);
        for (var i = 0; i < c; i++) if (d[i] == 0f) throw new DivideByZeroException($"Fusion denominator zero at row {i}.");

        var target = query.LossTarget;
        var g = new float[c];
        var softmaxF = MaskedFusion.Weights(f.AsSpan());
        for (var i = 0; i < c; i++) g[i] = softmaxF[i] - target[i];
        trainLoss = ListwiseLoss(f, target);

        runtime.Upload(Gbuf, g.AsSpan());
        runtime.SetBuffer("fusion_wgrad", 0, Sbuf);
        runtime.SetBuffer("fusion_wgrad", 1, Pbuf);
        runtime.SetBuffer("fusion_wgrad", 2, Gbuf);
        runtime.SetBuffer("fusion_wgrad", 3, Fbuf);
        runtime.SetBuffer("fusion_wgrad", 4, Dbuf);
        runtime.SetBuffer("fusion_wgrad", 5, thetaGradBuf);
        runtime.SetUInt32("fusion_wgrad", 6, (uint)c);
        runtime.Run("fusion_wgrad", new nuint[] { 4 }, new nuint[] { 1 });
        var contribution = runtime.Download(thetaGradBuf, TrainingSetBuilder.ViewCount);

        var dot = 0f;
        for (var i = 0; i < contribution.Length; i++) dot += w[i] * contribution[i];
        var dlGate = new float[TrainingSetBuilder.ViewCount];
        for (var i = 0; i < contribution.Length; i++) dlGate[i] = w[i] * (contribution[i] - dot);

        for (var k = 0; k < TrainingSetBuilder.ViewCount; k++)
        {
            gradB[k] += dlGate[k];
            for (var j = 0; j < features; j++) gradW[k * features + j] += dlGate[k] * x[j];
        }
    }

    void ApplyGlobalUpdate(float lr, float weightDecay, float[] gradTheta, double trainLoss)
    {
        adamStep++;
        var bc1 = (float)(1f - Math.Pow(0.9, adamStep));
        var bc2 = (float)(1f - Math.Pow(0.999, adamStep));
        // Convert gradient to device buffer.
        runtime.Upload(thetaGradBuf, gradTheta.AsSpan());
        runtime.SetBuffer("adam_update", 0, thetaBuf);
        runtime.SetBuffer("adam_update", 1, thetaGradBuf);
        runtime.SetBuffer("adam_update", 2, thetaM);
        runtime.SetBuffer("adam_update", 3, thetaV);
        runtime.SetFloat32("adam_update", 4, lr);
        runtime.SetFloat32("adam_update", 5, bc1);
        runtime.SetFloat32("adam_update", 6, bc2);
        runtime.SetFloat32("adam_update", 7, 1e-8f);
        runtime.SetFloat32("adam_update", 8, weightDecay);
        runtime.SetUInt32("adam_update", 9, (uint)theta.Length);
        runtime.Run("adam_update", new nuint[] { (nuint)RoundUp64(theta.Length) }, new nuint[] { 64 });
        runtime.Finish();
    }

    void ApplyGateUpdate(float lr, float weightDecay, float[] gradW, float[] gradB, double trainLoss)
    {
        adamStep++;
        var bc1 = (float)(1f - Math.Pow(0.9, adamStep));
        var bc2 = (float)(1f - Math.Pow(0.999, adamStep));
        runtime.Upload(gateGradWBuf, gradW.AsSpan());
        runtime.SetBuffer("adam_update", 0, gateWBuf);
        runtime.SetBuffer("adam_update", 1, gateGradWBuf);
        runtime.SetBuffer("adam_update", 2, gateWm);
        runtime.SetBuffer("adam_update", 3, gateWv);
        runtime.SetFloat32("adam_update", 4, lr);
        runtime.SetFloat32("adam_update", 5, bc1);
        runtime.SetFloat32("adam_update", 6, bc2);
        runtime.SetFloat32("adam_update", 7, 1e-8f);
        runtime.SetFloat32("adam_update", 8, weightDecay);
        runtime.SetUInt32("adam_update", 9, (uint)gateWeight.Length);
        runtime.Run("adam_update", new nuint[] { (nuint)RoundUp64(gateWeight.Length) }, new nuint[] { 64 });
        runtime.Finish();

        runtime.Upload(gateGradBBuf, gradB.AsSpan());
        runtime.SetBuffer("adam_update", 0, gateBBuf);
        runtime.SetBuffer("adam_update", 1, gateGradBBuf);
        runtime.SetBuffer("adam_update", 2, gateBm);
        runtime.SetBuffer("adam_update", 3, gateBv);
        runtime.SetFloat32("adam_update", 4, lr);
        runtime.SetFloat32("adam_update", 5, bc1);
        runtime.SetFloat32("adam_update", 6, bc2);
        runtime.SetFloat32("adam_update", 7, 1e-8f);
        runtime.SetFloat32("adam_update", 8, weightDecay);
        runtime.SetUInt32("adam_update", 9, (uint)gateBias.Length);
        runtime.Run("adam_update", new nuint[] { (nuint)RoundUp64(gateBias.Length) }, new nuint[] { 64 });
        runtime.Finish();
    }

    double MeanNdcg(IReadOnlyList<TrainingQuery> validation, int topK)
    {
        var total = 0d; var count = 0;
        foreach (var query in validation)
        {
            if (!query.IsUsable) continue;
            var fused = Scores(query);
            var ranked = RetrievalEvaluation.Rank(fused, topK);
            total += RetrievalEvaluation.Measure(ranked, query, topK).Ndcg;
            count++;
        }
        return count == 0 ? double.NaN : total / count;
    }

    static double ListwiseLoss(float[] f, IReadOnlyList<float> target)
    {
        var max = f.Max();
        var sum = f.Select(v => MathF.Exp(v - max)).Sum();
        var lnZ = max + MathF.Log(sum);
        var loss = 0d;
        for (var i = 0; i < f.Length; i++) loss -= target[i] * (f[i] - lnZ);
        return loss;
    }

    static TrainingQuery[] Usable(IReadOnlyList<TrainingQuery> queries) => [.. queries.Where(q => q.IsUsable)];

    static int RoundUp64(int n) => ((n + 63) / 64) * 64;

    public void Dispose() => runtime.Dispose();
}
