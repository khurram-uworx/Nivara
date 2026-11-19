using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AI;

/// <summary>
/// A small classification head that sits on top of a frozen ModernBERT encoder:
/// it takes the encoder's <c>[CLS]</c> vector (shape <c>[hiddenSize]</c> or
/// <c>[batch, hiddenSize]</c>) and produces class logits. Only this head's
/// parameters are meant to be trained; the encoder stays frozen.
/// </summary>
public sealed class ModernBertClassificationHead<TModel, TWeight>
    : Module<TModel>
    where TModel : struct, IFloatingPointIeee754<TModel>
    where TWeight : struct, IFloatingPointIeee754<TWeight>
{
    public readonly int HiddenSize;
    public readonly int NumClasses;
    public readonly Linear<TModel, TWeight> Hidden;
    public readonly Linear<TModel, TWeight> Classifier;

    public ModernBertClassificationHead(int hiddenSize, int numClasses, bool skipWeightInitialization = false)
    {
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (numClasses <= 0) throw new ArgumentOutOfRangeException(nameof(numClasses));

        HiddenSize = hiddenSize;
        NumClasses = numClasses;

        if (skipWeightInitialization)
        {
            Hidden = Linear<TModel, TWeight>.Uninitialized(hiddenSize, hiddenSize, true);
            Classifier = Linear<TModel, TWeight>.Uninitialized(hiddenSize, numClasses, true);
        }
        else
        {
            Hidden = new Linear<TModel, TWeight>(hiddenSize, hiddenSize, bias: true);
            Classifier = new Linear<TModel, TWeight>(hiddenSize, numClasses, bias: true);
        }

        RegisterModules(Hidden, Classifier);
    }

    public override ReverseGradTensor<TModel> Forward(ReverseGradTensor<TModel> input)
    {
        ArgumentNullException.ThrowIfNull(input);
        var h = ReverseGradOperations.GeluExact(Hidden.Forward(input));
        return Classifier.Forward(h);
    }

    /// <summary>Loads head weights from a safetensors-style dictionary keyed <c>head.*</c>.</summary>
    public static ModernBertClassificationHead<TModel, TWeight> LoadWeights(
        Dictionary<string, (TWeight[] Data, int[] Shape)> tensors,
        int hiddenSize,
        int numClasses,
        bool strict = false)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var head = new ModernBertClassificationHead<TModel, TWeight>(hiddenSize, numClasses, skipWeightInitialization: true);

        // Use StateDictLoader for Linear weights/bias.
        Nivara.AI.StateDictLoader.LoadLinear(head.Hidden, tensors, "head.hidden", strict);
        Nivara.AI.StateDictLoader.LoadLinear(head.Classifier, tensors, "head.classifier", strict);

        head.Eval();
        return head;
    }
}
