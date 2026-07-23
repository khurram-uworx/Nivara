using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Nn;

public sealed class TransformerBlock<T> : Module<T> where T : struct, INumber<T>
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

    public TransformerBlock(int nEmbd, int nHead, double dropout = 0.0, int maxSeqLen = 256, double initStd = 0.02)
    {
        if (nEmbd % nHead != 0)
            throw new ArgumentException($"nEmbd ({nEmbd}) must be divisible by nHead ({nHead}).");

        this.nEmbd = nEmbd;
        this.nHead = nHead;
        headDim = nEmbd / nHead;
        attnScale = T.CreateChecked(1.0 / Math.Sqrt(headDim));
        this.maxSeqLen = maxSeqLen;

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

        causalMask = CreateCausalMask(maxSeqLen);
    }

    ReverseGradTensor<T> CreateCausalMask(int maxLen)
    {
        var maskData = new T[maxLen * maxLen];
        for (int i = 0; i < maxLen; i++)
            for (int j = 0; j < maxLen; j++)
                if (j > i)
                    maskData[i * maxLen + j] = T.CreateChecked(double.NegativeInfinity);

        var col = NivaraColumn<T>.Create(maskData);
        var tensor = new ReverseGradTensor<T>(col, requiresGrad: false);
        tensor.Reshape(maxLen, maxLen);
        return tensor;
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
        var xNorm = PerRowRMSNorm(input, L, D);

        var Q = qProj.Forward(xNorm);
        var K = kProj.Forward(xNorm);
        var V = vProj.Forward(xNorm);

        var xAttn = MultiHeadAttention(Q, K, V, L);
        var xProj = oProj.Forward(xAttn);

        var oDrop = attnDropout != null ? attnDropout.Forward(xProj) : xProj;
        var x = ReverseGradOperations.Add(oDrop, xResidual);

        xResidual = x;
        var xMLPNorm = PerRowRMSNorm(x, L, D);

        var mlp1 = mlpFc1.Forward(xMLPNorm);
        var relu = ReverseGradOperations.Relu(mlp1);
        var squared = ReverseGradOperations.Multiply(relu, relu);
        var mlp2 = mlpFc2.Forward(squared);

        var mlpDrop = residualDropout != null ? residualDropout.Forward(mlp2) : mlp2;
        x = ReverseGradOperations.Add(mlpDrop, xResidual);

        return x;
    }

    ReverseGradTensor<T> MultiHeadAttention(ReverseGradTensor<T> Q, ReverseGradTensor<T> K,
        ReverseGradTensor<T> V, int L)
    {
        var heads = new ReverseGradTensor<T>[nHead];

        var scaleTensor = GradientUtils.Full(L * L, attnScale);
        scaleTensor.Reshape(L, L);

        var causalMask = CausalMaskSlice(L);

        for (int h = 0; h < nHead; h++)
        {
            int hs = h * headDim;

            var Q_h = ReverseGradOperations.Slice(Q, hs, headDim);
            var K_h = ReverseGradOperations.Slice(K, hs, headDim);
            var V_h = ReverseGradOperations.Slice(V, hs, headDim);

            var K_h_T = ReverseGradOperations.Transpose(K_h);
            var scores = ReverseGradOperations.MatMul(Q_h, K_h_T);

            scores = ReverseGradOperations.Multiply(scores, scaleTensor);

            scores = ReverseGradOperations.Add(scores, causalMask);

            var weights = ReverseGradOperations.Softmax(scores);

            heads[h] = ReverseGradOperations.MatMul(weights, V_h);
        }

        return ReverseGradOperations.Concat(heads, axis: 1);
    }

    static ReverseGradTensor<T> PerRowRMSNorm(ReverseGradTensor<T> x, int rows, int cols, double eps = 1e-5)
    {
        var srcData = new T[x.Length];
        x.Data.CopyTo(srcData, default(T)!);

        var resultData = new T[rows * cols];

        if (typeof(T) == typeof(float))
        {
            var srcFloat = System.Runtime.CompilerServices.Unsafe.As<T[], float[]>(ref srcData);
            var resFloat = System.Runtime.CompilerServices.Unsafe.As<T[], float[]>(ref resultData);

            for (int i = 0; i < rows; i++)
            {
                int baseIdx = i * cols;
                var row = srcFloat.AsSpan(baseIdx, cols);
                var dst = resFloat.AsSpan(baseIdx, cols);

                float sumSq = TensorPrimitives.Dot(row, row);
                float rms = MathF.Sqrt(sumSq / cols + (float)eps);
                float invRms = 1.0f / rms;
                TensorPrimitives.Multiply(row, invRms, dst);
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var srcDouble = System.Runtime.CompilerServices.Unsafe.As<T[], double[]>(ref srcData);
            var resDouble = System.Runtime.CompilerServices.Unsafe.As<T[], double[]>(ref resultData);

            for (int i = 0; i < rows; i++)
            {
                int baseIdx = i * cols;
                var row = srcDouble.AsSpan(baseIdx, cols);
                var dst = resDouble.AsSpan(baseIdx, cols);

                double sumSq = TensorPrimitives.Dot(row, row);
                double rms = Math.Sqrt(sumSq / cols + eps);
                double invRms = 1.0 / rms;
                TensorPrimitives.Multiply(row, invRms, dst);
            }
        }
        else
        {
            for (int i = 0; i < rows; i++)
            {
                int baseIdx = i * cols;

                double sumSq = 0;
                for (int j = 0; j < cols; j++)
                {
                    double v = double.CreateChecked(srcData[baseIdx + j]);
                    sumSq += v * v;
                }
                double rms = Math.Sqrt(sumSq / cols + eps);
                double invRms = 1.0 / rms;

                for (int j = 0; j < cols; j++)
                    resultData[baseIdx + j] = T.CreateChecked(double.CreateChecked(srcData[baseIdx + j]) * invRms);
            }
        }

        var resultCol = NivaraColumn<T>.Create(resultData);
        var result = new ReverseGradTensor<T>(resultCol, x.RequiresGrad, x.Shape);

        if (x.RequiresGrad)
        {
            var savedInput = new T[x.Length];
            x.Data.CopyTo(savedInput, default(T)!);

            var gradFn = new OpNode<T>("PerRowRMSNorm", [x], (typedGradOutput, sgn) =>
            {
                var gradOut = new T[typedGradOutput.Length];
                typedGradOutput.CopyTo(gradOut.AsSpan(), default(T)!);

                var gradResult = new T[rows * cols];

                for (int i = 0; i < rows; i++)
                {
                    int baseIdx = i * cols;

                    double sumSq = 0;
                    for (int j = 0; j < cols; j++)
                    {
                        double v = double.CreateChecked(savedInput[baseIdx + j]);
                        sumSq += v * v;
                    }
                    double rms = Math.Sqrt(sumSq / cols + eps);
                    double invRms = 1.0 / rms;
                    double rms3 = rms * rms * rms;

                    double sumGradX = 0;
                    for (int j = 0; j < cols; j++)
                    {
                        double g = double.CreateChecked(gradOut[baseIdx + j]);
                        double v = double.CreateChecked(savedInput[baseIdx + j]);
                        sumGradX += g * v;
                    }

                    double scale = sumGradX / (cols * rms3);

                    for (int j = 0; j < cols; j++)
                    {
                        double g = double.CreateChecked(gradOut[baseIdx + j]);
                        double v = double.CreateChecked(savedInput[baseIdx + j]);
                        gradResult[baseIdx + j] = T.CreateChecked(g * invRms - v * scale);
                    }
                }

                var gradCol = NivaraColumn<T>.Create(gradResult);
                ReverseGradOperations.AccumulateGradient(x, gradCol, sgn);
            });

            ComputationGraph.AddNode(result, gradFn);
        }

        return result;
    }
}
