using Nivara.AI;

namespace Nivara.Silk;

/// <summary>How a device-resident weight maps back to the checkpoint's <c>[out, in]</c> form.</summary>
internal enum SilkWeightLayout
{
    /// <summary>Device and checkpoint agree: embedding tables, LayerNorm γ/β, plain biases.</summary>
    Plain,
    /// <summary>Device is <c>[in, out]</c>; the checkpoint is the same matrix as <c>[out, in]</c>.</summary>
    Transposed,
    /// <summary>Device is one <c>[in, 3·out]</c> q/k/v block; the checkpoint has three <c>[out, in]</c>.</summary>
    PackedQkvWeight,
    /// <summary>Device is one <c>[3·out]</c> q/k/v bias; the checkpoint has three <c>[out]</c>.</summary>
    PackedQkvBias,
}

/// <summary>
/// A weight's checkpoint identity: the key(s) it came from, how the device layout relates to the
/// checkpoint's, and the checkpoint shape. <see cref="Keys"/> is empty for a buffer that is derived
/// on the device rather than loaded (the token-type row), which is why such a buffer is not exported.
/// </summary>
internal sealed record SilkWeightSlot(string Name, string[] Keys, SilkWeightLayout Layout, int[] Shape, bool NarrowMatrix = false)
{
    /// <summary>The element count of the device buffer.</summary>
    public int Count => Layout switch
    {
        SilkWeightLayout.PackedQkvWeight => 3 * Shape[0] * Shape[1],
        SilkWeightLayout.PackedQkvBias => 3 * Shape[0],
        _ => Shape.Length == 1 ? Shape[0] : Shape[0] * Shape[1],
    };
}

/// <summary>
/// One parameter on the device: its value buffer always, and — when the weight set is trainable —
/// the gradient and the two AdamW moments. <see cref="Slot"/> carries everything the exporter needs.
/// </summary>
internal sealed class SilkParam(SilkRuntime.SilkBuffer weight, SilkWeightSlot slot)
{
    public SilkRuntime.SilkBuffer Weight { get; } = weight;
    public SilkWeightSlot Slot { get; } = slot;

    public SilkRuntime.SilkBuffer Gradient { get; set; }
    public SilkRuntime.SilkBuffer FirstMoment { get; set; }
    public SilkRuntime.SilkBuffer SecondMoment { get; set; }

    /// <summary>
    /// Whether this parameter participates in backward accumulation and the AdamW step. Frozen
    /// parameters keep their value buffer but never receive a gradient, so their weights are constant.
    /// </summary>
    public bool Trainable { get; set; } = true;

    public nint Handle => Weight.Handle;
    public int Count => Weight.Capacity;
}

/// <summary>One layer's weights, in <see cref="WeightLayout"/>'s device layout.</summary>
internal sealed class SilkLayerWeights
{
    public required SilkParam Wqkv { get; init; }
    public required SilkParam Bqkv { get; init; }
    public required SilkParam O { get; init; }
    public required SilkParam W1 { get; init; }
    public required SilkParam W2 { get; init; }
    public required SilkParam Bo { get; init; }
    public required SilkParam B1 { get; init; }
    public required SilkParam B2 { get; init; }
    public required SilkParam Ln1W { get; init; }
    public required SilkParam Ln1B { get; init; }
    public required SilkParam Ln2W { get; init; }
    public required SilkParam Ln2B { get; init; }
}

/// <summary>
/// The BERT-family encoder's weights uploaded to the Silk device — the authoritative upload, layout
/// and checkpoint-mapping for every consumer that needs them, so the forward runner and the trainer
/// cannot drift into uploading the same checkpoint differently.
///
/// <para>
/// <b>Layout.</b> Matrix weights are uploaded transposed to device <c>[in, out]</c> and q/k/v is
/// packed to one <c>[in, 3·out]</c> block, both through <see cref="WeightLayout"/> — the same
/// arithmetic the ILGPU runner uses. <see cref="Export"/> inverts exactly that, so handing the
/// result to <c>DistilBertForSequenceClassification.LoadWeights</c> accepts what the device learned.
/// </para>
/// <para>
/// <b>Training.</b> With <c>trainable</c> set, every tracked parameter also gets a gradient buffer
/// and the two AdamW moment buffers, all zeroed. Buffers are owned and released by this type.
/// </para>
/// </summary>
internal sealed class SilkBertWeights : IDisposable
{
    const string Backend = "Silk";

    const string PreWKey = "pre_classifier.weight";
    const string PreBKey = "pre_classifier.bias";
    const string ClsWKey = "classifier.weight";
    const string ClsBKey = "classifier.bias";

    readonly SilkRuntime runtime;
    readonly Dictionary<string, (float[] Data, int[] Shape)> tensors;
    readonly bool trainable;
    readonly int trainableLayerCount;
    readonly bool narrowBf16;
    readonly bool narrowFp16;
    readonly List<nint> owned = [];
    readonly List<SilkParam> tracked;

    public SilkBertWeights(
        SilkRuntime runtime,
        BertConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        BertGpuNaming naming,
        bool trainable,
        int trainableLayerCount = -1)
    {
        this.runtime = runtime;
        this.tensors = tensors;
        this.trainable = trainable;
        this.narrowBf16 = false;
        this.narrowFp16 = false;
        tracked = new List<SilkParam>(capacity: 4 + 12 * config.NumHiddenLayers + 4);

        int normalizedTrainable = trainableLayerCount < 0
            ? (trainable ? config.NumHiddenLayers : 0)
            : SilkPartialTrainingPolicy.Normalize(trainableLayerCount, config.NumHiddenLayers, trainable);
        this.trainableLayerCount = normalizedTrainable;

        string emb = BertGpuKeys.Embeddings(naming);
        WordEmb = Plain("wordEmb", $"{emb}.word_embeddings.weight");
        PosEmb = Plain("posEmb", $"{emb}.position_embeddings.weight");
        EmbLnW = Plain("embLnW", $"{emb}.LayerNorm.weight");
        EmbLnB = Plain("embLnB", $"{emb}.LayerNorm.bias");

        string tokenTypeKey = $"{emb}.token_type_embeddings.weight";
        HasTokenType = tensors.ContainsKey(tokenTypeKey);
        var tokenType = HasTokenType
            ? WeightLayout.Req(tensors, tokenTypeKey, Backend).Data.AsSpan(0, config.HiddenSize).ToArray()
            : new float[config.HiddenSize];
        TokenTypeEmb = NewParam(
            new SilkWeightSlot("tokenTypeEmb", [], SilkWeightLayout.Plain, [config.HiddenSize]), tokenType, track: false);

        Layers = new SilkLayerWeights[config.NumHiddenLayers];
        for (int i = 0; i < Layers.Length; i++)
        {
            Layers[i] = UploadLayer(i, naming);
            bool layerTrainable = SilkPartialTrainingPolicy.LayerTrainable(i, config.NumHiddenLayers, normalizedTrainable, trainable);
            Layers[i].Wqkv.Trainable = layerTrainable;
            Layers[i].Bqkv.Trainable = layerTrainable;
            Layers[i].O.Trainable = layerTrainable;
            Layers[i].W1.Trainable = layerTrainable;
            Layers[i].W2.Trainable = layerTrainable;
            Layers[i].Bo.Trainable = layerTrainable;
            Layers[i].B1.Trainable = layerTrainable;
            Layers[i].B2.Trainable = layerTrainable;
            Layers[i].Ln1W.Trainable = layerTrainable;
            Layers[i].Ln1B.Trainable = layerTrainable;
            Layers[i].Ln2W.Trainable = layerTrainable;
            Layers[i].Ln2B.Trainable = layerTrainable;
        }

        bool embeddingsTrainable = SilkPartialTrainingPolicy.EmbeddingsTrainable(normalizedTrainable, config.NumHiddenLayers, trainable);
        WordEmb.Trainable = embeddingsTrainable;
        PosEmb.Trainable = embeddingsTrainable;
        EmbLnW.Trainable = embeddingsTrainable;
        EmbLnB.Trainable = embeddingsTrainable;

        HasHead = tensors.ContainsKey(PreWKey) && tensors.ContainsKey(ClsWKey);
        NumClasses = HasHead ? WeightLayout.Req(tensors, ClsWKey, Backend).Shape[0] : 0;
        PreW = Transposed("preW", PreWKey, required: HasHead);
        PreB = Plain("preB", PreBKey, required: HasHead);
        ClsW = Transposed("clsW", ClsWKey, required: HasHead);
        ClsB = Plain("clsB", ClsBKey, required: HasHead);
        if (HasHead)
        {
            bool headTrainable = SilkPartialTrainingPolicy.HeadTrainable(trainable);
            PreW.Trainable = headTrainable;
            PreB.Trainable = headTrainable;
            ClsW.Trainable = headTrainable;
            ClsB.Trainable = headTrainable;
        }
    }

    public SilkBertWeights(
        SilkRuntime runtime,
        BertConfig config,
        Dictionary<string, (ushort[] Data, int[] Shape)> tensors,
        BertGpuNaming naming,
        bool trainable,
        bool isBf16,
        bool isFp16,
        int trainableLayerCount = -1)
    {
        this.runtime = runtime;
        this.trainable = trainable;
        this.narrowBf16 = isBf16;
        this.narrowFp16 = isFp16;
        this.tensors = null!;

        this.trainableLayerCount = trainableLayerCount < 0
            ? (trainable ? config.NumHiddenLayers : 0)
            : SilkPartialTrainingPolicy.Normalize(trainableLayerCount, config.NumHiddenLayers, trainable);
        tracked = new List<SilkParam>(capacity: 4 + 12 * config.NumHiddenLayers + 4);

        Func<ushort, float> widen = isBf16
            ? (u => BitConverter.UInt32BitsToSingle((uint)u << 16))
            : (u => (float)BitConverter.Int16BitsToHalf((short)u));

        string emb = BertGpuKeys.Embeddings(naming);
        WordEmb = PlainUshort("wordEmb", $"{emb}.word_embeddings.weight", widen, tensors);
        PosEmb = PlainUshort("posEmb", $"{emb}.position_embeddings.weight", widen, tensors);
        EmbLnW = PlainUshort("embLnW", $"{emb}.LayerNorm.weight", widen, tensors);
        EmbLnB = PlainUshort("embLnB", $"{emb}.LayerNorm.bias", widen, tensors);

        string tokenTypeKey = $"{emb}.token_type_embeddings.weight";
        HasTokenType = tensors.ContainsKey(tokenTypeKey);
        ushort[] tokenTypeSrc = HasTokenType
            ? WeightLayout.Req(tensors, tokenTypeKey, Backend).Data.AsSpan(0, config.HiddenSize).ToArray()
            : new ushort[config.HiddenSize];
        var tokenType = new float[config.HiddenSize];
        for (int i = 0; i < tokenType.Length; i++) tokenType[i] = widen(tokenTypeSrc[i]);
        TokenTypeEmb = NewParam(
            new SilkWeightSlot("tokenTypeEmb", [], SilkWeightLayout.Plain, [config.HiddenSize]), tokenType, track: false);

        Layers = new SilkLayerWeights[config.NumHiddenLayers];
        for (int i = 0; i < Layers.Length; i++)
        {
            Layers[i] = UploadLayerNarrow(i, naming, isBf16, isFp16, widen, tensors);
            bool layerTrainable = SilkPartialTrainingPolicy.LayerTrainable(i, config.NumHiddenLayers, this.trainableLayerCount, trainable);
            Layers[i].Wqkv.Trainable = layerTrainable;
            Layers[i].Bqkv.Trainable = layerTrainable;
            Layers[i].O.Trainable = layerTrainable;
            Layers[i].W1.Trainable = layerTrainable;
            Layers[i].W2.Trainable = layerTrainable;
            Layers[i].Bo.Trainable = layerTrainable;
            Layers[i].B1.Trainable = layerTrainable;
            Layers[i].B2.Trainable = layerTrainable;
            Layers[i].Ln1W.Trainable = layerTrainable;
            Layers[i].Ln1B.Trainable = layerTrainable;
            Layers[i].Ln2W.Trainable = layerTrainable;
            Layers[i].Ln2B.Trainable = layerTrainable;
        }

        bool embeddingsTrainable2 = SilkPartialTrainingPolicy.EmbeddingsTrainable(this.trainableLayerCount, config.NumHiddenLayers, trainable);
        WordEmb.Trainable = embeddingsTrainable2;
        PosEmb.Trainable = embeddingsTrainable2;
        EmbLnW.Trainable = embeddingsTrainable2;
        EmbLnB.Trainable = embeddingsTrainable2;

        HasHead = tensors.ContainsKey(PreWKey) && tensors.ContainsKey(ClsWKey);
        NumClasses = HasHead ? WeightLayout.Req(tensors, ClsWKey, Backend).Shape[0] : 0;
        PreW = HasHead
            ? TransposedNarrowMat("preW", PreWKey, tensors)
            : NewParam(new SilkWeightSlot("preW", [PreWKey], SilkWeightLayout.Transposed, [1, 1], NarrowMatrix: false), [0f], track: false);
        PreB = PlainUshort("preB", PreBKey, widen, tensors, required: HasHead);
        ClsW = HasHead
            ? TransposedNarrowMat("clsW", ClsWKey, tensors)
            : NewParam(new SilkWeightSlot("clsW", [ClsWKey], SilkWeightLayout.Transposed, [1, 1], NarrowMatrix: false), [0f], track: false);
        ClsB = PlainUshort("clsB", ClsBKey, widen, tensors, required: HasHead);
        if (HasHead)
        {
            bool headTrainable = SilkPartialTrainingPolicy.HeadTrainable(trainable);
            PreW.Trainable = headTrainable;
            PreB.Trainable = headTrainable;
            ClsW.Trainable = headTrainable;
            ClsB.Trainable = headTrainable;
        }
    }

    public SilkParam WordEmb { get; }
    public SilkParam PosEmb { get; }
    public SilkParam EmbLnW { get; }
    public SilkParam EmbLnB { get; }
    public SilkParam TokenTypeEmb { get; }
    public SilkLayerWeights[] Layers { get; }
    public SilkParam PreW { get; }
    public SilkParam PreB { get; }
    public SilkParam ClsW { get; }
    public SilkParam ClsB { get; }

    public bool HasHead { get; }
    public bool HasTokenType { get; }
    public int NumClasses { get; }

    public bool IsNarrowBf16 => narrowBf16;
    public bool IsNarrowFp16 => narrowFp16;

    /// <summary>Every loaded parameter, in a stable order: embeddings, then per layer, then the
    /// head. Derived buffers (the token-type row) are excluded.</summary>
    public IReadOnlyList<SilkParam> Parameters => tracked;

    /// <summary>
    /// Downloads every checkpoint key this set was loaded from and inverts the upload layout:
    /// transposed matrices become <c>[out, in]</c>, the packed q/k/v block is split back into three
    /// tensors, and the plain buffers pass through. Shapes match what <c>LoadWeights</c> expects
    /// (biases are 1-D). The token-type row is derived and therefore absent.
    /// </summary>
    public Dictionary<string, (float[] Data, int[] Shape)> Export() => ExportCore(p => p.Weight, narrowWeights: true);

    /// <summary>
    /// The same inversion as <see cref="Export"/>, over the accumulated gradients instead of the
    /// weights: every key <see cref="Export"/> produces appears here too, in the checkpoint's own
    /// shape, so a device gradient can be compared with a reverse-mode pass on the CPU model
    /// key-for-key.
    /// </summary>
    public Dictionary<string, (float[] Data, int[] Shape)> ExportGradients()
    {
        if (!trainable)
            throw new InvalidOperationException("This weight set was uploaded read-only and has no gradients.");
        // Gradients are always stored in F32, even for narrow (BF16/FP16) weight matrices, so they
        // are downloaded as floats regardless of the weight storage type.
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
                Func<ushort, float> widen = narrowBf16
                    ? (u => BitConverter.UInt32BitsToSingle((uint)u << 16))
                    : (u => (float)BitConverter.Int16BitsToHalf((short)u));
                var us = runtime.DownloadUshort(select(p).Handle, p.Count);
                device = new float[us.Length];
                for (int i = 0; i < device.Length; i++) device[i] = widen(us[i]);
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

    SilkLayerWeights UploadLayer(int index, BertGpuNaming naming)
    {
        string q = BertGpuKeys.Attention(naming, index, 'q');
        string k = BertGpuKeys.Attention(naming, index, 'k');
        string v = BertGpuKeys.Attention(naming, index, 'v');
        string o = BertGpuKeys.Attention(naming, index, 'o');
        string ffn1 = BertGpuKeys.Ffn(naming, index, 1);
        string ffn2 = BertGpuKeys.Ffn(naming, index, 2);
        string ln1 = BertGpuKeys.LayerNorm(naming, index, 1);
        string ln2 = BertGpuKeys.LayerNorm(naming, index, 2);

        var (wqkv, bqkv) = WeightLayout.QkvPacked(tensors, q, k, v, Backend);
        int[] qkvShape = WeightLayout.Req(tensors, $"{q}.weight", Backend).Shape;
        int outDim = qkvShape[0], inDim = qkvShape[1];

        string layer = $"layer{index}";
        var wqkvSlot = new SilkWeightSlot(
            $"{layer}.Wqkv", [$"{q}.weight", $"{k}.weight", $"{v}.weight"],
            SilkWeightLayout.PackedQkvWeight, [outDim, inDim]);
        var bqkvSlot = new SilkWeightSlot(
            $"{layer}.Bqkv", [$"{q}.bias", $"{k}.bias", $"{v}.bias"],
            SilkWeightLayout.PackedQkvBias, [outDim]);

        return new SilkLayerWeights
        {
            Wqkv = NewParam(wqkvSlot, wqkv),
            Bqkv = NewParam(bqkvSlot, bqkv),
            O = Transposed($"{layer}.O", $"{o}.weight"),
            W1 = Transposed($"{layer}.W1", $"{ffn1}.weight"),
            W2 = Transposed($"{layer}.W2", $"{ffn2}.weight"),
            Bo = Plain($"{layer}.Bo", $"{o}.bias"),
            B1 = Plain($"{layer}.B1", $"{ffn1}.bias"),
            B2 = Plain($"{layer}.B2", $"{ffn2}.bias"),
            Ln1W = Plain($"{layer}.Ln1W", $"{ln1}.weight"),
            Ln1B = Plain($"{layer}.Ln1B", $"{ln1}.bias"),
            Ln2W = Plain($"{layer}.Ln2W", $"{ln2}.weight"),
            Ln2B = Plain($"{layer}.Ln2B", $"{ln2}.bias"),
        };
    }

    SilkParam Plain(string name, string key, bool required = true)
    {
        var slot = new SilkWeightSlot(name, [key], SilkWeightLayout.Plain,
            required ? WeightLayout.Req(tensors, key, Backend).Shape : [1]);
        return required
            ? NewParam(slot, WeightLayout.Req(tensors, key, Backend).Data)
            : NewParam(slot, [0f], track: false);
    }

    SilkParam Transposed(string name, string key, bool required = true)
    {
        var slot = new SilkWeightSlot(name, [key], SilkWeightLayout.Transposed,
            required ? WeightLayout.Req(tensors, key, Backend).Shape : [1, 1]);
        return required
            ? NewParam(slot, WeightLayout.Transposed(tensors, key, Backend))
            : NewParam(slot, [0f], track: false);
    }

    SilkParam NewParam(SilkWeightSlot slot, ReadOnlySpan<float> data, bool track = true)
    {
        if (data.Length != slot.Count)
            throw new InvalidOperationException(
                $"Silk weight '{slot.Name}' has {data.Length} elements but its {slot.Layout} slot expects {slot.Count}.");

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
            throw new InvalidOperationException(
                $"Silk weight '{slot.Name}' has {data.Length} elements but its {slot.Layout} slot expects {slot.Count}.");

        nint buffer = Register(runtime.AllocateBytes((ulong)data.Length * sizeof(ushort)));
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

    SilkParam PlainUshort(string name, string key, Func<ushort, float> widen,
        Dictionary<string, (ushort[] Data, int[] Shape)> tensors, bool required = true)
    {
        var slot = new SilkWeightSlot(name, [key], SilkWeightLayout.Plain,
            required ? WeightLayout.Req(tensors, key, Backend).Shape : [1], NarrowMatrix: false);
        if (!required)
        {
            return tensors.ContainsKey(key)
                ? PlainUshortCore(slot, WeightLayout.Req(tensors, key, Backend).Data, widen)
                : NewParam(slot, [0f], track: false);
        }
        return PlainUshortCore(slot, WeightLayout.Req(tensors, key, Backend).Data, widen);
    }

    SilkParam PlainUshortCore(SilkWeightSlot slot, ushort[] data, Func<ushort, float> widen)
    {
        var floats = new float[data.Length];
        for (int i = 0; i < floats.Length; i++) floats[i] = widen(data[i]);
        return NewParam(slot, floats);
    }

    SilkLayerWeights UploadLayerNarrow(int index, BertGpuNaming naming, bool isBf16, bool isFp16,
        Func<ushort, float> widen, Dictionary<string, (ushort[] Data, int[] Shape)> tensors)
    {
        string q = BertGpuKeys.Attention(naming, index, 'q');
        string k = BertGpuKeys.Attention(naming, index, 'k');
        string v = BertGpuKeys.Attention(naming, index, 'v');
        string o = BertGpuKeys.Attention(naming, index, 'o');
        string ffn1 = BertGpuKeys.Ffn(naming, index, 1);
        string ffn2 = BertGpuKeys.Ffn(naming, index, 2);
        string ln1 = BertGpuKeys.LayerNorm(naming, index, 1);
        string ln2 = BertGpuKeys.LayerNorm(naming, index, 2);

        var (wqkv, bqkv) = WeightLayout.QkvPackedNarrow(tensors, q, k, v, widen, Backend);
        int[] qkvShape = WeightLayout.Req(tensors, $"{q}.weight", Backend).Shape;
        int outDim = qkvShape[0], inDim = qkvShape[1];

        string layer = $"layer{index}";
        var wqkvSlot = new SilkWeightSlot(
            $"{layer}.Wqkv", [$"{q}.weight", $"{k}.weight", $"{v}.weight"],
            SilkWeightLayout.PackedQkvWeight, [outDim, inDim], NarrowMatrix: true);
        var bqkvSlot = new SilkWeightSlot(
            $"{layer}.Bqkv", [$"{q}.bias", $"{k}.bias", $"{v}.bias"],
            SilkWeightLayout.PackedQkvBias, [outDim], NarrowMatrix: false);

        var bqkvFloat = bqkv;

        return new SilkLayerWeights
        {
            Wqkv = NewParamNarrow(wqkvSlot, wqkv),
            Bqkv = NewParam(bqkvSlot, bqkvFloat),
            O = TransposedNarrowMat($"layer{index}.O", $"{o}.weight", tensors),
            W1 = TransposedNarrowMat($"{layer}.W1", $"{ffn1}.weight", tensors),
            W2 = TransposedNarrowMat($"{layer}.W2", $"{ffn2}.weight", tensors),
            Bo = PlainUshort($"{layer}.Bo", $"{o}.bias", widen, tensors),
            B1 = PlainUshort($"{layer}.B1", $"{ffn1}.bias", widen, tensors),
            B2 = PlainUshort($"{layer}.B2", $"{ffn2}.bias", widen, tensors),
            Ln1W = PlainUshort($"{layer}.Ln1W", $"{ln1}.weight", widen, tensors),
            Ln1B = PlainUshort($"{layer}.Ln1B", $"{ln1}.bias", widen, tensors),
            Ln2W = PlainUshort($"{layer}.Ln2W", $"{ln2}.weight", widen, tensors),
            Ln2B = PlainUshort($"{layer}.Ln2B", $"{ln2}.bias", widen, tensors),
        };
    }

    SilkParam TransposedNarrowMat(string name, string key, Dictionary<string, (ushort[] Data, int[] Shape)> tensors)
    {
        var slot = new SilkWeightSlot(name, [key], SilkWeightLayout.Transposed,
            WeightLayout.Req(tensors, key, Backend).Shape, NarrowMatrix: true);
        return NewParamNarrow(slot, WeightLayout.Transposed(tensors, key, Backend));
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
}
