using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using System.Numerics;

namespace Nivara.Samples;

public static class StateDictLoader
{
    public static void LoadEmbed<TModel, TWeight>(
        Embedding<TModel> embed,
        Dictionary<string, (TWeight[] Data, int[] Shape)> tensors,
        string key)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        if (!tensors.TryGetValue(key, out var t))
            throw new KeyNotFoundException($"Missing tensor: {key}");
        var tensor = TypeConverter.Convert<TWeight, TModel>(
            ReverseGradTensor<TWeight>.FromMatrix(t.Data, t.Shape[0], t.Shape[1]));
        embed.LoadStateDict(new Dictionary<string, ReverseGradTensor<TModel>> { ["Weight"] = tensor });
    }

    public static void LoadLinear<TModel, TWeight>(
        Linear<TModel> linear,
        Dictionary<string, (TWeight[] Data, int[] Shape)> tensors,
        string prefix)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        var dict = new Dictionary<string, ReverseGradTensor<TModel>>();
        if (tensors.TryGetValue($"{prefix}.weight", out var w))
            dict["Weight"] = TypeConverter.Convert<TWeight, TModel>(
                ReverseGradTensor<TWeight>.FromMatrix(w.Data, w.Shape[0], w.Shape[1]));
        if (tensors.TryGetValue($"{prefix}.bias", out var b))
            dict["Bias"] = TypeConverter.Convert<TWeight, TModel>(
                ReverseGradTensor<TWeight>.FromMatrix(b.Data, 1, b.Shape[0]));
        if (dict.Count > 0) linear.LoadStateDict(dict);
    }

    /// <summary>
    /// Loads one contiguous row block out of a fused weight into a <see cref="Linear{T}"/>. ModernBERT
    /// fuses its QKV projections into a single <c>Wqkv</c> and its MLP gate/up projections into a
    /// single <c>Wi</c>; each target projection is a row slice of the fused matrix because
    /// <see cref="Linear{T}"/> stores its weight as <c>[outFeatures, inFeatures]</c>.
    /// </summary>
    /// <param name="rowOffset">Index of the first row of the block within the fused weight.</param>
    /// <param name="rowCount">Number of rows in the block.</param>
    public static void LoadLinearSlice<TModel, TWeight>(
        Linear<TModel> linear,
        Dictionary<string, (TWeight[] Data, int[] Shape)> tensors,
        string prefix,
        int rowOffset,
        int rowCount)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        ArgumentNullException.ThrowIfNull(linear);

        if (rowOffset < 0)
            throw new ArgumentOutOfRangeException(nameof(rowOffset), $"rowOffset must be non-negative, got {rowOffset}.");
        if (rowCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(rowCount), $"rowCount must be positive, got {rowCount}.");

        if (!tensors.TryGetValue($"{prefix}.weight", out var w))
            throw new KeyNotFoundException($"Missing tensor: {prefix}.weight");

        if (w.Shape.Length != 2)
            throw new InvalidOperationException(
                $"Expected a 2D weight at {prefix}.weight to slice, got rank {w.Shape.Length}.");

        int inFeatures = w.Shape[1];
        int fusedRows = w.Shape[0];
        if (rowOffset + rowCount > fusedRows)
            throw new InvalidOperationException(
                $"Row block [{rowOffset}, {rowOffset + rowCount}) exceeds the {fusedRows} rows of {prefix}.weight.");
        if (inFeatures != linear.InFeatures || rowCount != linear.OutFeatures)
            throw new InvalidOperationException(
                $"Row block of {prefix}.weight is [{rowCount}, {inFeatures}] but the target linear is " +
                $"[{linear.OutFeatures}, {linear.InFeatures}].");

        var slice = new TWeight[rowCount * inFeatures];
        Array.Copy(w.Data, rowOffset * inFeatures, slice, 0, slice.Length);

        var tensor = TypeConverter.Convert<TWeight, TModel>(
            ReverseGradTensor<TWeight>.FromMatrix(slice, rowCount, inFeatures));
        linear.LoadStateDict(new Dictionary<string, ReverseGradTensor<TModel>> { ["Weight"] = tensor });
    }

    /// <summary>
    /// Loads one contiguous row block of a fused weight and its matching bias block into a
    /// <see cref="Linear{T}"/>. The fused-QKV counterpart of <see cref="LoadLinearSlice{TModel, TWeight}"/>
    /// for checkpoints that name the two tensors themselves rather than deriving them from a prefix.
    /// </summary>
    /// <remarks>
    /// The two naming conventions in the wild are genuinely different, which is why this takes keys
    /// and not a prefix. ModernBERT writes <c>…attn.Wqkv.weight</c>, which
    /// <see cref="LoadLinearSlice{TModel, TWeight}"/>'s prefix-plus-<c>.weight</c> convention
    /// already covers, and it has no bias. PyTorch's <c>nn.MultiheadAttention</c> writes
    /// <c>…self_attn.in_proj_weight</c> and <c>…self_attn.in_proj_bias</c> — an underscore before the
    /// role, not a dot — and both exist. A prefix helper would silently look up
    /// <c>in_proj.weight</c>, find nothing, and throw a message naming a key the checkpoint does not
    /// contain.
    /// </remarks>
    /// <param name="rowOffset">Index of the first row of the weight block within the fused weight.</param>
    /// <param name="rowCount">Number of rows in the block, and of the bias block.</param>
    public static void LoadFusedLinearSlice<TModel, TWeight>(
        Linear<TModel> linear,
        Dictionary<string, (TWeight[] Data, int[] Shape)> tensors,
        string weightKey,
        string biasKey,
        int rowOffset,
        int rowCount)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        ArgumentNullException.ThrowIfNull(linear);

        if (rowOffset < 0)
            throw new ArgumentOutOfRangeException(nameof(rowOffset), $"rowOffset must be non-negative, got {rowOffset}.");
        if (rowCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(rowCount), $"rowCount must be positive, got {rowCount}.");

        if (!tensors.TryGetValue(weightKey, out var w))
            throw new KeyNotFoundException($"Missing tensor: {weightKey}");
        if (!tensors.TryGetValue(biasKey, out var b))
            throw new KeyNotFoundException($"Missing tensor: {biasKey}");

        if (w.Shape.Length != 2)
            throw new InvalidOperationException(
                $"Expected a 2D weight at {weightKey} to slice, got rank {w.Shape.Length}.");
        if (b.Shape.Length != 1)
            throw new InvalidOperationException(
                $"Expected a 1D bias at {biasKey} to slice, got rank {b.Shape.Length}.");

        int inFeatures = w.Shape[1];
        int fusedRows = w.Shape[0];
        if (rowOffset + rowCount > fusedRows)
            throw new InvalidOperationException(
                $"Row block [{rowOffset}, {rowOffset + rowCount}) exceeds the {fusedRows} rows of {weightKey}.");
        if (rowOffset + rowCount > b.Shape[0])
            throw new InvalidOperationException(
                $"Block [{rowOffset}, {rowOffset + rowCount}) exceeds the {b.Shape[0]} entries of {biasKey}.");
        if (inFeatures != linear.InFeatures || rowCount != linear.OutFeatures)
            throw new InvalidOperationException(
                $"Row block of {weightKey} is [{rowCount}, {inFeatures}] but the target linear is " +
                $"[{linear.OutFeatures}, {linear.InFeatures}].");
        if (b.Shape[0] != w.Shape[0])
            throw new InvalidOperationException(
                $"{weightKey} has {w.Shape[0]} rows but {biasKey} has {b.Shape[0]}; they are one fused block.");

        var state = new Dictionary<string, ReverseGradTensor<TModel>>();

        var weightSlice = new TWeight[rowCount * inFeatures];
        Array.Copy(w.Data, rowOffset * inFeatures, weightSlice, 0, weightSlice.Length);
        state["Weight"] = TypeConverter.Convert<TWeight, TModel>(
            ReverseGradTensor<TWeight>.FromMatrix(weightSlice, rowCount, inFeatures));

        var biasSlice = new TWeight[rowCount];
        Array.Copy(b.Data, rowOffset, biasSlice, 0, rowCount);
        state["Bias"] = TypeConverter.Convert<TWeight, TModel>(
            ReverseGradTensor<TWeight>.FromMatrix(biasSlice, 1, rowCount));

        linear.LoadStateDict(state);
    }

    public static void LoadLayerNorm<TModel, TWeight>(
        LayerNorm<TModel> ln,
        Dictionary<string, (TWeight[] Data, int[] Shape)> tensors,
        string prefix)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        var dict = new Dictionary<string, ReverseGradTensor<TModel>>();
        if (tensors.TryGetValue($"{prefix}.weight", out var w))
            dict["Weight"] = TypeConverter.Convert<TWeight, TModel>(
                ReverseGradTensor<TWeight>.FromArray(w.Data));
        if (tensors.TryGetValue($"{prefix}.bias", out var b))
            dict["Bias"] = TypeConverter.Convert<TWeight, TModel>(
                ReverseGradTensor<TWeight>.FromArray(b.Data));
        if (dict.Count > 0) ln.LoadStateDict(dict);
    }

    public static void LoadRMSNorm<TModel, TWeight>(
        RMSNorm<TModel> rms,
        Dictionary<string, (TWeight[] Data, int[] Shape)> tensors,
        string prefix)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        if (!tensors.TryGetValue($"{prefix}.weight", out var w))
            throw new KeyNotFoundException($"Missing tensor: {prefix}.weight");
        var tensor = TypeConverter.Convert<TWeight, TModel>(
            ReverseGradTensor<TWeight>.FromArray(w.Data));
        rms.LoadStateDict(new Dictionary<string, ReverseGradTensor<TModel>> { ["Weight"] = tensor });
    }
}
