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
