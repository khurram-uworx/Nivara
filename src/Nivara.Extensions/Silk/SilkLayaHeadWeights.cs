using Nivara.AI;

namespace Nivara.Silk;

internal sealed class SilkLayaHeadWeights : IDisposable
{
    const string Backend = "Silk";

    readonly SilkRuntime runtime;
    readonly Dictionary<string, (float[] Data, int[] Shape)> floatTensors;
    readonly Dictionary<string, (ushort[] Data, int[] Shape)> narrowTensors;
    readonly bool trainable;
    readonly bool narrowBf16;
    readonly bool narrowFp16;
    readonly int actClassCount;
    readonly List<nint> owned = [];
    readonly List<SilkParam> tracked;

    public SilkLayaHeadWeights(
        SilkRuntime runtime,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        int headLayerCount,
        int actClassCount,
        bool trainable)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(tensors);
        this.runtime = runtime;
        this.floatTensors = tensors;
        this.narrowTensors = null!;
        this.trainable = trainable;
        this.narrowBf16 = false;
        this.narrowFp16 = false;
        this.actClassCount = actClassCount;
        tracked = new List<SilkParam>(capacity: 16 + 12 * headLayerCount);

        HiddenSize = WeightLayout.Req(tensors, "type_emb.weight", Backend).Shape[1];
        HeadLayerCount = headLayerCount;
        ActHidden = LayaDecisionHead<float>.ActHeadHidden;

        TypeEmb = Plain("typeEmb", "type_emb.weight");

        ScorerNormW = Plain("scorerNormW", "scorer.0.weight");
        ScorerNormB = Plain("scorerNormB", "scorer.0.bias");
        ScorerHiddenW = Transposed("scorerHiddenW", "scorer.1.weight");
        ScorerHiddenB = Plain("scorerHiddenB", "scorer.1.bias");
        ScorerOutW = Transposed("scorerOutW", "scorer.3.weight");
        ScorerOutB = Plain("scorerOutB", "scorer.3.bias");

        ActInW = Transposed("actInW", "act_head.0.weight");
        ActInB = Plain("actInB", "act_head.0.bias");
        ActOutW = Transposed("actOutW", "act_head.2.weight");
        ActOutB = Plain("actOutB", "act_head.2.bias");

        Layers = new LayerWeights[headLayerCount];
        for (int i = 0; i < headLayerCount; i++)
            Layers[i] = UploadLayer(i);

        foreach (var p in Parameters)
            p.Trainable = SilkPartialTrainingPolicy.HeadTrainable(trainable);
    }

    public SilkLayaHeadWeights(
        SilkRuntime runtime,
        Dictionary<string, (ushort[] Data, int[] Shape)> tensors,
        int headLayerCount,
        int actClassCount,
        bool trainable,
        bool isBf16,
        bool isFp16)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(tensors);
        this.runtime = runtime;
        this.floatTensors = null!;
        this.narrowTensors = tensors;
        this.trainable = trainable;
        this.narrowBf16 = isBf16;
        this.narrowFp16 = isFp16;
        this.actClassCount = actClassCount;
        tracked = new List<SilkParam>(capacity: 16 + 12 * headLayerCount);

        HiddenSize = WeightLayout.Req(tensors, "type_emb.weight", Backend).Shape[1];
        HeadLayerCount = headLayerCount;
        ActHidden = LayaDecisionHead<float>.ActHeadHidden;

        Func<ushort, float> widen = isBf16 ? SilkRuntime.WidenBf16 : SilkRuntime.WidenFp16;

        TypeEmb = PlainUshort("typeEmb", "type_emb.weight", widen);

        ScorerNormW = PlainUshort("scorerNormW", "scorer.0.weight", widen);
        ScorerNormB = PlainUshort("scorerNormB", "scorer.0.bias", widen);
        ScorerHiddenW = TransposedNarrowMat("scorerHiddenW", "scorer.1.weight", tensors);
        ScorerHiddenB = PlainUshort("scorerHiddenB", "scorer.1.bias", widen);
        ScorerOutW = TransposedNarrowMat("scorerOutW", "scorer.3.weight", tensors);
        ScorerOutB = PlainUshort("scorerOutB", "scorer.3.bias", widen);

        ActInW = TransposedNarrowMat("actInW", "act_head.0.weight", tensors);
        ActInB = PlainUshort("actInB", "act_head.0.bias", widen);
        ActOutW = TransposedNarrowMat("actOutW", "act_head.2.weight", tensors);
        ActOutB = PlainUshort("actOutB", "act_head.2.bias", widen);

        Layers = new LayerWeights[headLayerCount];
        for (int i = 0; i < headLayerCount; i++)
            Layers[i] = UploadLayerNarrow(i, widen);

        foreach (var p in Parameters)
            p.Trainable = SilkPartialTrainingPolicy.HeadTrainable(trainable);
    }

    public int HiddenSize { get; }
    public int HeadLayerCount { get; }
    public int ActHidden { get; }
    public int ActClassCount => actClassCount;

    public SilkParam TypeEmb { get; }
    public SilkParam ScorerNormW { get; }
    public SilkParam ScorerNormB { get; }
    public SilkParam ScorerHiddenW { get; }
    public SilkParam ScorerHiddenB { get; }
    public SilkParam ScorerOutW { get; }
    public SilkParam ScorerOutB { get; }
    public SilkParam ActInW { get; }
    public SilkParam ActInB { get; }
    public SilkParam ActOutW { get; }
    public SilkParam ActOutB { get; }
    public LayerWeights[] Layers { get; }

    public IReadOnlyList<SilkParam> Parameters => tracked;

    public Dictionary<string, (float[] Data, int[] Shape)> Export()
        => ExportCore(p => p.Weight, narrowWeights: true);

    public Dictionary<string, (float[] Data, int[] Shape)> ExportGradients()
    {
        if (!trainable) throw new InvalidOperationException("Weight set was uploaded read-only.");
        return ExportCore(p => p.Gradient, narrowWeights: false);
    }

    Dictionary<string, (float[] Data, int[] Shape)> ExportCore(Func<SilkParam, SilkRuntime.SilkBuffer> select, bool narrowWeights)
    {
        var result = new Dictionary<string, (float[] Data, int[] Shape)>(StringComparer.Ordinal);
        foreach (var p in tracked)
        {
            if (p.Slot.Keys.Length == 0) continue;
            float[] device;
            if (narrowWeights && p.Slot.NarrowMatrix)
            {
                if (narrowBf16)
                {
                    var bf = runtime.DownloadBFloat16(select(p).Handle, p.Count);
                    device = new float[bf.Length];
                    for (int i = 0; i < device.Length; i++) device[i] = (float)bf[i];
                }
                else
                {
                    var us = runtime.DownloadUshort(select(p).Handle, p.Count);
                    device = new float[us.Length];
                    for (int i = 0; i < device.Length; i++) device[i] = SilkRuntime.WidenFp16(us[i]);
                }
            }
            else
            {
                device = runtime.Download(select(p).Handle, p.Count);
            }
            switch (p.Slot.Layout)
            {
                case SilkWeightLayout.Plain:
                    result[p.Slot.Keys[0]] = (device, p.Slot.Shape);
                    break;
                case SilkWeightLayout.Transposed:
                    {
                        int outDim = p.Slot.Shape[0], inDim = p.Slot.Shape[1];
                        var data = new float[outDim * inDim];
                        for (int r = 0; r < outDim; r++)
                            for (int c = 0; c < inDim; c++)
                                data[r * inDim + c] = device[c * outDim + r];
                        result[p.Slot.Keys[0]] = (data, [outDim, inDim]);
                        break;
                    }
                case SilkWeightLayout.PackedQkvWeight:
                    {
                        int outDim = p.Slot.Shape[0], inDim = p.Slot.Shape[1], concat = 3 * outDim;
                        for (int block = 0; block < 3; block++)
                        {
                            var data = new float[outDim * inDim];
                            for (int r = 0; r < outDim; r++)
                                for (int c = 0; c < inDim; c++)
                                    data[r * inDim + c] = device[c * concat + block * outDim + r];
                            result[p.Slot.Keys[block]] = (data, [outDim, inDim]);
                        }
                        break;
                    }
                case SilkWeightLayout.PackedQkvBias:
                    {
                        int outDim = p.Slot.Shape[0];
                        for (int block = 0; block < 3; block++)
                            result[p.Slot.Keys[block]] =
                                (device.AsSpan(block * outDim, outDim).ToArray(), [outDim]);
                        break;
                    }
            }
        }
        return result;
    }

    public void Dispose()
    {
        foreach (nint buffer in owned)
            runtime.Release(buffer);
        owned.Clear();
    }

    // ---- F32 helpers ----

    SilkParam Plain(string name, string key)
    {
        var slot = new SilkWeightSlot(name, [key], SilkWeightLayout.Plain,
            WeightLayout.Req(floatTensors, key, Backend).Shape);
        return NewParam(slot, WeightLayout.Req(floatTensors, key, Backend).Data);
    }

    SilkParam Transposed(string name, string key)
    {
        var slot = new SilkWeightSlot(name, [key], SilkWeightLayout.Transposed,
            WeightLayout.Req(floatTensors, key, Backend).Shape);
        return NewParam(slot, WeightLayout.Transposed(floatTensors, key, Backend));
    }

    LayerWeights UploadLayer(int index)
    {
        string prefix = $"head.layers.{index}";
        string attn = $"{prefix}.self_attn";
        int hidden = HiddenSize;
        int interm = 4 * hidden;

        // Fused in_proj: checkpoint weight [3*hidden, hidden], bias [3*hidden].
        var inProj = WeightLayout.Req(floatTensors, $"{attn}.in_proj_weight", Backend);
        var inProjBias = WeightLayout.Req(floatTensors, $"{attn}.in_proj_bias", Backend);
        int outDim = inProj.Shape[0] / 3; // hidden
        int inDim = inProj.Shape[1]; // hidden
        int concat = 3 * outDim;
        var wqkv = new float[inDim * concat];
        for (int block = 0; block < 3; block++)
            for (int r = 0; r < outDim; r++)
                for (int c = 0; c < inDim; c++)
                    wqkv[c * concat + block * outDim + r] = inProj.Data[(block * outDim + r) * inDim + c];
        var bqkv = inProjBias.Data.ToArray();

        var wqkvSlot = new SilkWeightSlot(
            $"{prefix}.Wqkv", [$"{attn}.in_proj_weight"],
            SilkWeightLayout.PackedQkvWeight, [outDim, inDim]);
        var bqkvSlot = new SilkWeightSlot(
            $"{prefix}.Bqkv", [$"{attn}.in_proj_bias"],
            SilkWeightLayout.PackedQkvBias, [outDim]);

        // Norms
        var norm1W = Plain($"{prefix}.norm1W", $"{prefix}.norm1.weight");
        var norm1B = Plain($"{prefix}.norm1B", $"{prefix}.norm1.bias");
        var norm2W = Plain($"{prefix}.norm2W", $"{prefix}.norm2.weight");
        var norm2B = Plain($"{prefix}.norm2B", $"{prefix}.norm2.bias");

        // out_proj
        var woSlot = new SilkWeightSlot($"{prefix}.Wo", [$"{attn}.out_proj.weight"], SilkWeightLayout.Transposed, [hidden, hidden]);
        var wo = NewParam(woSlot, WeightLayout.Transposed(floatTensors, $"{attn}.out_proj.weight", Backend));
        var bo = Plain($"{prefix}.Bo", $"{attn}.out_proj.bias");

        // linear1
        var w1Slot = new SilkWeightSlot($"{prefix}.W1", [$"{prefix}.linear1.weight"], SilkWeightLayout.Transposed, [interm, hidden]);
        var w1 = NewParam(w1Slot, WeightLayout.Transposed(floatTensors, $"{prefix}.linear1.weight", Backend));
        var b1 = Plain($"{prefix}.B1", $"{prefix}.linear1.bias");

        // linear2
        var w2Slot = new SilkWeightSlot($"{prefix}.W2", [$"{prefix}.linear2.weight"], SilkWeightLayout.Transposed, [hidden, interm]);
        var w2 = NewParam(w2Slot, WeightLayout.Transposed(floatTensors, $"{prefix}.linear2.weight", Backend));
        var b2 = Plain($"{prefix}.B2", $"{prefix}.linear2.bias");

        return new LayerWeights
        {
            Norm1W = norm1W,
            Norm1B = norm1B,
            Wqkv = NewParam(wqkvSlot, wqkv),
            Bqkv = NewParam(bqkvSlot, bqkv),
            Wo = wo,
            Bo = bo,
            Norm2W = norm2W,
            Norm2B = norm2B,
            W1 = w1,
            B1 = b1,
            W2 = w2,
            B2 = b2,
        };
    }

    // ---- Narrow helpers ----

    SilkParam PlainUshort(string name, string key, Func<ushort, float> widen)
    {
        var t = WeightLayout.Req(narrowTensors, key, Backend);
        var slot = new SilkWeightSlot(name, [key], SilkWeightLayout.Plain, t.Shape, NarrowMatrix: false);
        var floats = new float[t.Data.Length];
        for (int i = 0; i < floats.Length; i++) floats[i] = widen(t.Data[i]);
        return NewParam(slot, floats);
    }

    SilkParam TransposedNarrowMat(string name, string key, Dictionary<string, (ushort[] Data, int[] Shape)> tensors)
    {
        var slot = new SilkWeightSlot(name, [key], SilkWeightLayout.Transposed,
            WeightLayout.Req(tensors, key, Backend).Shape, NarrowMatrix: true);
        return NewParamNarrow(slot, WeightLayout.Transposed(tensors, key, Backend));
    }

    LayerWeights UploadLayerNarrow(int index, Func<ushort, float> widen)
    {
        string prefix = $"head.layers.{index}";
        string attn = $"{prefix}.self_attn";
        int hidden = HiddenSize;
        int interm = 4 * hidden;

        var inProj = WeightLayout.Req(narrowTensors, $"{attn}.in_proj_weight", Backend);
        var inProjBias = WeightLayout.Req(narrowTensors, $"{attn}.in_proj_bias", Backend);
        int outDim = inProj.Shape[0] / 3;
        int inDim = inProj.Shape[1];
        int concat = 3 * outDim;
        var wqkv = new ushort[inDim * concat];
        for (int block = 0; block < 3; block++)
            for (int r = 0; r < outDim; r++)
                for (int c = 0; c < inDim; c++)
                    wqkv[c * concat + block * outDim + r] = inProj.Data[(block * outDim + r) * inDim + c];

        var bqkvFloat = new float[concat];
        for (int i = 0; i < concat; i++) bqkvFloat[i] = widen(inProjBias.Data[i]);

        var wqkvSlot = new SilkWeightSlot(
            $"{prefix}.Wqkv", [$"{attn}.in_proj_weight"],
            SilkWeightLayout.PackedQkvWeight, [outDim, inDim], NarrowMatrix: true);
        var bqkvSlot = new SilkWeightSlot(
            $"{prefix}.Bqkv", [$"{attn}.in_proj_bias"],
            SilkWeightLayout.PackedQkvBias, [outDim], NarrowMatrix: false);

        var norm1W = PlainUshort($"{prefix}.norm1W", $"{prefix}.norm1.weight", widen);
        var norm1B = PlainUshort($"{prefix}.norm1B", $"{prefix}.norm1.bias", widen);
        var norm2W = PlainUshort($"{prefix}.norm2W", $"{prefix}.norm2.weight", widen);
        var norm2B = PlainUshort($"{prefix}.norm2B", $"{prefix}.norm2.bias", widen);

        var woSlot = new SilkWeightSlot($"{prefix}.Wo", [$"{attn}.out_proj.weight"], SilkWeightLayout.Transposed, [hidden, hidden], NarrowMatrix: true);
        var wo = NewParamNarrow(woSlot, WeightLayout.Transposed(narrowTensors, $"{attn}.out_proj.weight", Backend));

        var bo = PlainUshort($"{prefix}.Bo", $"{attn}.out_proj.bias", widen);

        var w1Slot = new SilkWeightSlot($"{prefix}.W1", [$"{prefix}.linear1.weight"], SilkWeightLayout.Transposed, [interm, hidden], NarrowMatrix: true);
        var w1 = NewParamNarrow(w1Slot, WeightLayout.Transposed(narrowTensors, $"{prefix}.linear1.weight", Backend));

        var b1 = PlainUshort($"{prefix}.B1", $"{prefix}.linear1.bias", widen);

        var w2Slot = new SilkWeightSlot($"{prefix}.W2", [$"{prefix}.linear2.weight"], SilkWeightLayout.Transposed, [hidden, interm], NarrowMatrix: true);
        var w2 = NewParamNarrow(w2Slot, WeightLayout.Transposed(narrowTensors, $"{prefix}.linear2.weight", Backend));

        var b2 = PlainUshort($"{prefix}.B2", $"{prefix}.linear2.bias", widen);

        return new LayerWeights
        {
            Norm1W = norm1W,
            Norm1B = norm1B,
            Wqkv = NewParamNarrow(wqkvSlot, wqkv),
            Bqkv = NewParam(bqkvSlot, bqkvFloat),
            Wo = wo,
            Bo = bo,
            Norm2W = norm2W,
            Norm2B = norm2B,
            W1 = w1,
            B1 = b1,
            W2 = w2,
            B2 = b2,
        };
    }

    SilkParam NewParam(SilkWeightSlot slot, ReadOnlySpan<float> data, bool track = true)
    {
        if (data.Length != slot.Count)
            throw new InvalidOperationException($"Silk weight '{slot.Name}' has {data.Length} elements but slot expects {slot.Count}.");
        nint buffer = Register(runtime.Allocate(data.Length));
        runtime.Upload(buffer, data);
        var param = new SilkParam(new SilkRuntime.SilkBuffer(buffer, data.Length), slot);
        if (trainable && track)
        {
            param.Gradient = Zeroed(data.Length);
            param.FirstMoment = Zeroed(data.Length);
            param.SecondMoment = Zeroed(data.Length);
        }
        if (track) tracked.Add(param);
        return param;
    }

    SilkParam NewParamNarrow(SilkWeightSlot slot, ReadOnlySpan<ushort> data, bool track = true)
    {
        if (data.Length != slot.Count)
            throw new InvalidOperationException($"Silk weight '{slot.Name}' has {data.Length} elements but slot expects {slot.Count}.");
        nint buffer = Register(runtime.AllocateUshort(data.Length));
        runtime.Upload(buffer, data);
        var param = new SilkParam(new SilkRuntime.SilkBuffer(buffer, data.Length), slot);
        if (trainable && track)
        {
            param.Gradient = Zeroed(data.Length);
            param.FirstMoment = Zeroed(data.Length);
            param.SecondMoment = Zeroed(data.Length);
        }
        if (track) tracked.Add(param);
        return param;
    }

    SilkRuntime.SilkBuffer Zeroed(int count)
    {
        nint buffer = Register(runtime.Allocate(count));
        SilkLauncher.Fill(runtime, buffer, 0f, count);
        return new SilkRuntime.SilkBuffer(buffer, count);
    }

    nint Register(nint buffer)
    {
        owned.Add(buffer);
        return buffer;
    }

    // ---- Layer weights container ----

    public sealed class LayerWeights
    {
        public required SilkParam Norm1W { get; init; }
        public required SilkParam Norm1B { get; init; }
        public required SilkParam Wqkv { get; init; }
        public required SilkParam Bqkv { get; init; }
        public required SilkParam Wo { get; init; }
        public required SilkParam Bo { get; init; }
        public required SilkParam Norm2W { get; init; }
        public required SilkParam Norm2B { get; init; }
        public required SilkParam W1 { get; init; }
        public required SilkParam B1 { get; init; }
        public required SilkParam W2 { get; init; }
        public required SilkParam B2 { get; init; }
    }
}
