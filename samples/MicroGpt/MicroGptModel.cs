using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics;

namespace MicroGpt;

/// <summary>
/// MicroGPT — a miniature GPT language model running on Nivara's AutoDiff engine.
///
/// Faithful per-position port of Andrej Karpathy's microgpt.py:
///   Token Embedding → Position Embedding → N × Transformer Block → Output Projection
///
/// Each Transformer Block (per-position with KV cache):
///   RMSNorm → Multi-Head Self-Attention (per-head dot products) → Residual
///   RMSNorm → MLP (expand 4×, squared ReLU, compress) → Residual
///
/// Weight tying: output projection reuses the token embedding matrix (wte).
/// </summary>
public class MicroGptModel<T> : IDisposable where T : struct, INumber<T>
{
    readonly int nEmbd;
    readonly int nLayer;
    readonly int blockSize;
    readonly int nHead;
    readonly int headDim;
    readonly int vocabSize;

    // Shared scale factor for attention: 1/sqrt(headDim)
    readonly T attnScale;

    // Embeddings
    readonly Embedding<T> wte;  // token embeddings [vocabSize, nEmbd]
    readonly Embedding<T> wpe;  // position embeddings [blockSize, nEmbd]

    // Per-layer attention projections
    readonly List<Linear<T>> attnWq;
    readonly List<Linear<T>> attnWk;
    readonly List<Linear<T>> attnWv;
    readonly List<Linear<T>> attnWo;

    // Per-layer MLP
    readonly List<Linear<T>> mlpFc1;
    readonly List<Linear<T>> mlpFc2;

    // All parameters
    readonly List<Parameter<T>> allParams;

    readonly Linear<T>? lmHead;

    public MicroGptModel(int vocabSize, int nEmbd, int nLayer, int blockSize, int nHead,
        double initStd = 0.02, bool weightTying = true)
    {
        if (nEmbd % nHead != 0)
            throw new ArgumentException("nEmbd must be divisible by nHead");

        this.vocabSize = vocabSize;
        this.nEmbd = nEmbd;
        this.nLayer = nLayer;
        this.blockSize = blockSize;
        this.nHead = nHead;
        headDim = nEmbd / nHead;
        attnScale = T.CreateChecked(1.0 / Math.Sqrt(headDim));

        allParams = [];

        wte = CreateEmbedding("wte", vocabSize, nEmbd);
        wpe = CreateEmbedding("wpe", blockSize, nEmbd);

        lmHead = weightTying ? null : CreateLinear("lm_head", vocabSize, nEmbd, initStd);

        attnWq = []; attnWk = []; attnWv = []; attnWo = [];
        mlpFc1 = []; mlpFc2 = [];

        for (int i = 0; i < nLayer; i++)
        {
            attnWq.Add(CreateLinear($"{i}.attn_wq", nEmbd, nEmbd, initStd));
            attnWk.Add(CreateLinear($"{i}.attn_wk", nEmbd, nEmbd, initStd));
            attnWv.Add(CreateLinear($"{i}.attn_wv", nEmbd, nEmbd, initStd));
            attnWo.Add(CreateLinear($"{i}.attn_wo", nEmbd, nEmbd, 0.0));
            mlpFc1.Add(CreateLinear($"{i}.mlp_fc1", 4 * nEmbd, nEmbd, initStd));
            mlpFc2.Add(CreateLinear($"{i}.mlp_fc2", nEmbd, 4 * nEmbd, 0.0));
        }
    }

    Embedding<T> CreateEmbedding(string name, int num, int dim)
    {
        var emb = new Embedding<T>(num, dim);
        allParams.AddRange(emb.GetParameters().Values);
        return emb;
    }

    Linear<T> CreateLinear(string name, int outF, int inF, double std)
    {
        var lin = new Linear<T>(inF, outF, bias: false,
            weightInitializer: new Nivara.AutoDiff.Nn.Initializers.NormalInitializer<T>(
                T.Zero, T.CreateChecked(std)));
        allParams.Add(lin.WeightParam);
        return lin;
    }

    /// <summary>
    /// Forward pass for a single token position with KV cache.
    /// </summary>
    public ReverseGradTensor<T> Forward(
        int tokenId, int posId,
        List<ReverseGradTensor<T>>[] keys,
        List<ReverseGradTensor<T>>[] values)
    {
        // Token + position embedding
        var tokEmb = wte.Forward(tokenId);                           // [1, nEmbd]
        var posEmb = wpe.Forward(posId % blockSize);                 // [1, nEmbd]
        var x = ReverseGradOperations.Add(tokEmb, posEmb);           // [1, nEmbd]

        for (int li = 0; li < nLayer; li++)
        {
            // --- Multi-Head Self-Attention (per-position, per-head) ---
            var xResidual = x;
            x = ReverseGradOperations.RMSNorm(x);                     // [1, nEmbd]

            var q = attnWq[li].Forward(x);                            // [1, nEmbd]
            var k = attnWk[li].Forward(x);                            // [1, nEmbd]
            var v = attnWv[li].Forward(x);                            // [1, nEmbd]

            keys[li].Add(k);
            values[li].Add(v);

            var xAttnParts = new List<ReverseGradTensor<T>>();

            for (int h = 0; h < nHead; h++)
            {
                int hs = h * headDim;
                int cacheLen = keys[li].Count;

                // Slice Q for this head [1, headDim]
                var qHead = ReverseGradOperations.Slice(q, hs, headDim);

                // Per-position scores and value slices
                var scores = new List<ReverseGradTensor<T>>(cacheLen);
                var vSlices = new List<ReverseGradTensor<T>>(cacheLen);

                for (int t = 0; t < cacheLen; t++)
                {
                    var kHead = ReverseGradOperations.Slice(keys[li][t], hs, headDim);
                    var vHead = ReverseGradOperations.Slice(values[li][t], hs, headDim);

                    // score = q·k / sqrt(headDim) — differentiable dot product
                    var dot = ReverseGradOperations.Sum(
                        ReverseGradOperations.Multiply(qHead, kHead));  // [1]

                    scores.Add(ReverseGradOperations.Multiply(dot, Scalar(attnScale)));
                    vSlices.Add(vHead);
                }

                // Softmax over positions (all [1] tensors → list of [1])
                var weights = SoftmaxList(scores);

                // Weighted blend: sum(weight[t] * v[t]) over positions
                ReverseGradTensor<T>? headOut = null;
                for (int t = 0; t < cacheLen; t++)
                {
                    // Broadcast weight [1] → [1, headDim]
                    var broadW = BroadcastScalar(weights[t], headDim);
                    var weightedV = ReverseGradOperations.Multiply(broadW, vSlices[t]);

                    headOut = headOut == null
                        ? weightedV
                        : ReverseGradOperations.Add(headOut, weightedV);
                }

                if (headOut != null)
                    xAttnParts.Add(headOut);
            }

            // Concatenate heads horizontally
            var xAttn = ConcatHeads(xAttnParts);                       // [1, nEmbd]

            // Output projection + residual
            x = attnWo[li].Forward(xAttn);                             // [1, nEmbd]
            x = ReverseGradOperations.Add(x, xResidual);                // [1, nEmbd]

            // --- MLP ---
            xResidual = x;
            x = ReverseGradOperations.RMSNorm(x);                     // [1, nEmbd]
            x = mlpFc1[li].Forward(x);                                 // [1, 4*nEmbd]
            var relu = ReverseGradOperations.Relu(x);
            x = ReverseGradOperations.Multiply(relu, relu);            // Squared ReLU [1, 4*nEmbd]
            x = mlpFc2[li].Forward(x);                                 // [1, nEmbd]
            x = ReverseGradOperations.Add(x, xResidual);                // [1, nEmbd]
        }

        // Output projection
        if (lmHead != null)
        {
            return lmHead.Forward(x);                                  // [1, vocabSize]
        }
        else
        {
            // Weight tying: reuse token embedding weight transposed
            var wteWeight = wte.Weight;
            var wteT = ReverseGradOperations.Transpose(wteWeight);     // [nEmbd, vocabSize]
            return ReverseGradOperations.MatMul(x, wteT);              // [1, vocabSize]
        }
    }

    /// <summary>
    /// Concatenate head outputs [1, headDim] × nHead → [1, nEmbd].
    /// Uses selection matrices for differentiable concatenation.
    /// </summary>
    static ReverseGradTensor<T> ConcatHeads(List<ReverseGradTensor<T>> heads)
    {
        if (heads.Count == 0)
            throw new ArgumentException("No heads to concatenate");
        if (heads.Count == 1)
            return heads[0];

        int totalDim = heads.Sum(h => h.Shape[1]);

        // Start with the first head, then use differentiable stacking
        var result = heads[0];  // [1, headDim0]
        int offset = heads[0].Shape[1];

        for (int i = 1; i < heads.Count; i++)
        {
            var h = heads[i];  // [1, headDim_i]
            int hDim = h.Shape[1];

            // To differentiable-stack: create a projection that maps
            // result [1, offset] + h [1, hDim] → [1, offset+hDim]
            // by using block-diagonal selection matrices.

            // Build selector for existing result: identity [offset, offset] placed at [0..offset-1, 0..offset-1]
            // Build selector for new head: identity [hDim, hDim] placed at [offset..offset+hDim, 0..hDim-1]
            // Combined: full selection matrix [offset+hDim, offset+hDim]

            var fullDim = offset + hDim;

            // Use forward concat via Sum(Multiply(...)) approach
            // Create result as: Concat(result, h) by a block matmul
            // result_part = result @ identity[offset, offset] — just identity
            // h_part = h placed at offset position
            // total = result_part + h_part_padded

            // Since result and h are row vectors [1, dim], we use Slice in reverse:
            // Output is [1, fullDim] where:
            //   output[0..offset] = result
            //   output[offset..fullDim] = h

            // Create output as sum of two contributions using selection matrices
            // Actually, use the identity: MatMul and Add
            // For result: identity_M1 [offset, fullDim] with diag at [:, 0:offset]
            // For h: identity_M2 [hDim, fullDim] with diag at [:, offset:fullDim]
            // result_part = Transpose(MatMul(M1_transposed))
            // Wait, this is circular.

            // Simplest correct approach: fill a zero tensor via Add of padded vectors
            // Padded result: expand from [1, offset] to [1, fullDim] with zeros at end
            // Padded h: expand from [1, hDim] to [1, fullDim] with zeros at start

            var paddedResult = PadRight(result, fullDim);  // [1, fullDim]
            var paddedH = PadLeft(h, fullDim);              // [1, fullDim]
            result = ReverseGradOperations.Add(paddedResult, paddedH);
            offset = fullDim;
        }

        return result;  // [1, nEmbd]
    }

    /// <summary>
    /// Differentiable padding: [1, dim] → [1, totalDim] with zeros appended.
    /// Uses a selection matrix extractor in reverse: output = result @ I_right.
    /// </summary>
    static ReverseGradTensor<T> PadRight(ReverseGradTensor<T> tensor, int totalDim)
    {
        int dim = tensor.Shape[1];
        // Build M [dim, totalDim]: identity placed at left [0:dim, 0:dim]
        var mData = new T[dim * totalDim];
        for (int i = 0; i < dim; i++)
            mData[i * totalDim + i] = T.One;

        var mCol = NivaraColumn<T>.Create(mData);
        var mTensor = new ReverseGradTensor<T>(mCol, requiresGrad: false);
        mTensor.Reshape(dim, totalDim);

        // tensor: [1, dim], mTensor: [dim, totalDim]
        // tM = tensor @ mTensor = MatMul(tensor, mTensor)
        // [1, dim] @ [dim, totalDim] = [1, totalDim]
        return ReverseGradOperations.MatMul(tensor, mTensor);
    }

    /// <summary>
    /// Differentiable padding: [1, dim] → [1, totalDim] with zeros prepended.
    /// Uses a selection matrix: identity placed at right [0:dim, totalDim-dim:totalDim].
    /// </summary>
    static ReverseGradTensor<T> PadLeft(ReverseGradTensor<T> tensor, int totalDim)
    {
        int dim = tensor.Shape[1];
        int offset = totalDim - dim;
        var mData = new T[dim * totalDim];
        for (int i = 0; i < dim; i++)
            mData[i * totalDim + offset + i] = T.One;

        var mCol = NivaraColumn<T>.Create(mData);
        var mTensor = new ReverseGradTensor<T>(mCol, requiresGrad: false);
        mTensor.Reshape(dim, totalDim);

        return ReverseGradOperations.MatMul(tensor, mTensor);
    }

    /// <summary>
    /// Broadcast a [1] scalar tensor to [1, dim] by MatMul with a ones vector.
    /// </summary>
    static ReverseGradTensor<T> BroadcastScalar(ReverseGradTensor<T> scalar, int dim)
    {
        var ones = GradientUtils.Ones<T>(dim);
        ones.Reshape(dim, 1);                          // [dim, 1]

        // scalar is [1], reshape to [1, 1]
        var s = scalar;
        if (s.Shape.Length < 2 || s.Shape[0] != 1 || s.Shape[1] != 1)
            s.Reshape(1, 1);

        // MatMul(ones [dim,1], s [1,1]) → [dim, 1], then Transpose → [1, dim]
        var broad = ReverseGradOperations.Transpose(
            ReverseGradOperations.MatMul(ones, s));
        return broad;
    }

    /// <summary>
    /// Wrap a scalar T value as a non-differentiable [1] tensor for arithmetic.
    /// </summary>
    static ReverseGradTensor<T> Scalar(T value)
    {
        var col = NivaraColumn<T>.Create([value]);
        return new ReverseGradTensor<T>(col, requiresGrad: false);
    }

    /// <summary>
    /// Differentiable softmax over a list of scalar [1] tensors.
    /// Returns list of [1] tensors (the softmax weights).
    /// </summary>
    static List<ReverseGradTensor<T>> SoftmaxList(List<ReverseGradTensor<T>> logits)
    {
        var exps = logits.Select(l => ReverseGradOperations.Exp(l)).ToList();
        var sumExp = exps[0];
        for (int i = 1; i < exps.Count; i++)
            sumExp = ReverseGradOperations.Add(sumExp, exps[i]);
        return exps.Select(e => ReverseGradOperations.Divide(e, sumExp)).ToList();
    }

    public IReadOnlyList<Parameter<T>> Parameters => allParams.AsReadOnly();

    public void Dispose()
    {
        foreach (var p in allParams)
            p.Dispose();
    }
}
