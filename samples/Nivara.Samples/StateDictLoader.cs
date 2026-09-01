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
