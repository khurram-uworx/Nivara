using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public enum NormType { RMSNorm, LayerNorm }

public sealed class TransformerBlock<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int nEmbd;
    readonly int nHead;
    readonly int headDim;
    readonly T attnScale;

    readonly Linear<T> qProj;
    readonly Linear<T> kProj;
    readonly Linear<T> vProj;
    readonly Linear<T> oProj;
    readonly Linear<T> mlpFc1;
    readonly Linear<T> mlpFc2;

    readonly Dropout<T>? attnDropout;
    readonly Dropout<T>? residualDropout;

    readonly ReverseGradTensor<T> causalMask;
    readonly int maxSeqLen;
    readonly NormType normType;

    public TransformerBlock(int nEmbd, int nHead, double dropout = 0.0, int maxSeqLen = 256, double initStd = 0.02, NormType normType = NormType.RMSNorm)
    {
        if (nEmbd % nHead != 0)
            throw new ArgumentException($"nEmbd ({nEmbd}) must be divisible by nHead ({nHead}).");

        this.nEmbd = nEmbd;
        this.nHead = nHead;
        headDim = nEmbd / nHead;
        attnScale = T.CreateChecked(1.0 / Math.Sqrt(headDim));
        this.maxSeqLen = maxSeqLen;
        this.normType = normType;

        var weightInit = new NormalInitializer<T>(T.Zero, T.CreateChecked(initStd));
        var zeroInit = new NormalInitializer<T>(T.Zero, T.Zero);

        qProj = new Linear<T>(nEmbd, nEmbd, bias: false, weightInitializer: weightInit);
        kProj = new Linear<T>(nEmbd, nEmbd, bias: false, weightInitializer: weightInit);
        vProj = new Linear<T>(nEmbd, nEmbd, bias: false, weightInitializer: weightInit);
        oProj = new Linear<T>(nEmbd, nEmbd, bias: false, weightInitializer: zeroInit);
        mlpFc1 = new Linear<T>(nEmbd, 4 * nEmbd, bias: false, weightInitializer: weightInit);
        mlpFc2 = new Linear<T>(4 * nEmbd, nEmbd, bias: false, weightInitializer: zeroInit);

        RegisterModules(qProj, kProj, vProj, oProj, mlpFc1, mlpFc2);

        if (dropout > 0.0)
        {
            attnDropout = new Dropout<T>(dropout);
            residualDropout = new Dropout<T>(dropout);
            RegisterModules(attnDropout, residualDropout);
        }

        causalMask = ModuleHelpers<T>.CreateCausalMask(maxSeqLen);
    }

    ReverseGradTensor<T> CausalMaskSlice(int L)
    {
        if (L >= maxSeqLen) return causalMask;
        var data = new T[L * L];
        var srcData = new T[causalMask.Length];
        causalMask.Data.CopyTo(srcData, default(T)!);
        for (int i = 0; i < L; i++)
            for (int j = 0; j < L; j++)
                data[i * L + j] = srcData[i * maxSeqLen + j];
        var col = NivaraColumn<T>.Create(data);
        var tensor = new ReverseGradTensor<T>(col, requiresGrad: false);
        tensor.Reshape(L, L);
        return tensor;
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        int L = input.shape[0];
        int D = input.shape.Length >= 2 ? input.shape[1] : nEmbd;

        var xResidual = input;
        var xNorm = ApplyNorm(input, L, D);

        var Q = qProj.Forward(xNorm);
        var K = kProj.Forward(xNorm);
        var V = vProj.Forward(xNorm);

        var xAttn = MultiHeadAttention(Q, K, V, L);
        var xProj = oProj.Forward(xAttn);

        var oDrop = attnDropout != null ? attnDropout.Forward(xProj) : xProj;
        var x = ReverseGradOperations.Add(oDrop, xResidual);

        xResidual = x;
        var xMLPNorm = ApplyNorm(x, L, D);

        var mlp1 = mlpFc1.Forward(xMLPNorm);
        var gelu = Activation.Gelu(mlp1);
        var mlp2 = mlpFc2.Forward(gelu);

        var mlpDrop = residualDropout != null ? residualDropout.Forward(mlp2) : mlp2;
        x = ReverseGradOperations.Add(mlpDrop, xResidual);

        return x;
    }

    ReverseGradTensor<T> MultiHeadAttention(ReverseGradTensor<T> Q, ReverseGradTensor<T> K,
        ReverseGradTensor<T> V, int L)
    {
        var scaleTensor = GradientUtils.Full(L * L, attnScale);
        scaleTensor.Reshape(L, L);

        var mask = CausalMaskSlice(L);

        return ModuleHelpers<T>.MultiHeadAttention(Q, K, V, nHead, headDim, scaleTensor, mask);
    }

    ReverseGradTensor<T> ApplyNorm(ReverseGradTensor<T> x, int rows, int cols)
    {
        return normType == NormType.RMSNorm
            ? PerRowRMSNorm(x, rows, cols)
            : PerRowLayerNorm(x, rows, cols);
    }

    static ReverseGradTensor<T> PerRowLayerNorm(ReverseGradTensor<T> x, int rows, int cols, double eps = 1e-5)
    {
        var srcData = new T[x.Length];
        x.Data.CopyTo(srcData, default(T)!);

        var result = LayerNormKernel<T>.Forward(
            srcData, rows, cols,
            ReadOnlySpan<T>.Empty, ReadOnlySpan<T>.Empty,
            T.CreateChecked(eps), affine: false);

        var resultCol = NivaraColumn<T>.CreateFromOwnedArray(result.Output);
        var outTensor = new ReverseGradTensor<T>(resultCol, x.RequiresGrad, x.Shape);

        if (x.RequiresGrad)
        {
            var gradFn = new OpNode<T>("PerRowLayerNorm", [x], (typedGradOutput) =>
            {
                var gradOut = new T[typedGradOutput.Length];
                typedGradOutput.CopyTo(gradOut.AsSpan(), default(T)!);

                var gradInput = LayerNormKernel<T>.BackwardInput(
                    gradOut, result.XHat,
                    ReadOnlySpan<T>.Empty, result.InvStd,
                    rows, cols, affine: false);

                var gradCol = NivaraColumn<T>.Create(gradInput);
                ReverseGradOperations.AccumulateGradient(x, gradCol);
            });

            ComputationGraph.AddNode(outTensor, gradFn);
        }

        return outTensor;
    }

    static ReverseGradTensor<T> PerRowRMSNorm(ReverseGradTensor<T> x, int rows, int cols, double eps = 1e-5)
    {
        var srcData = new T[x.Length];
        x.Data.CopyTo(srcData, default(T)!);

        var resultData = new T[rows * cols];

        RMSNormKernel<T>.PerRowRMSNormForwardKernel(srcData, resultData, rows, cols, eps);

        var resultCol = NivaraColumn<T>.Create(resultData);
        var result = new ReverseGradTensor<T>(resultCol, x.RequiresGrad, x.Shape);

        if (x.RequiresGrad)
        {
            var savedInput = new T[x.Length];
            x.Data.CopyTo(savedInput, default(T)!);

            var gradFn = new OpNode<T>("PerRowRMSNorm", [x], (typedGradOutput) =>
            {
                var gradOut = new T[typedGradOutput.Length];
                typedGradOutput.CopyTo(gradOut.AsSpan(), default(T)!);

                var gradResult = new T[rows * cols];

                RMSNormKernel<T>.PerRowRMSNormBackwardKernel(
                    savedInput, gradOut, gradResult, rows, cols, eps);

                var gradCol = NivaraColumn<T>.Create(gradResult);
                ReverseGradOperations.AccumulateGradient(x, gradCol);
            });

            ComputationGraph.AddNode(result, gradFn);
        }

        return result;
    }
}
