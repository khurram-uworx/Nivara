using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Variational autoencoder with optional conditioning. Encodes an input into latent mean and
/// log-variance, samples a latent via the reparameterization trick, and decodes it back to the
/// input space. Supports β-VAE weighting and KL-annealing loss modes.
/// </summary>
public sealed class VAE<T> : Module<T>, IMultipleInputModule<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly Linear<T> encoderLayer1;
    readonly Linear<T> encoderLayer2;
    readonly Linear<T> muHead;
    readonly Linear<T> logVarHead;
    readonly Linear<T> decoderLayer1;
    readonly Linear<T> decoderLayer2;
    readonly Parameter<T> beta;
    readonly Func<ReverseGradTensor<T>, ReverseGradTensor<T>> activation;
    readonly int inputDim;
    readonly int latentDim;
    readonly int hiddenDim;
    readonly int conditionDim;

    /// <summary>
    /// Creates a variational autoencoder.
    /// </summary>
    /// <param name="inputDim">Dimension of the input data (must be positive)</param>
    /// <param name="latentDim">Dimension of the latent space (must be positive)</param>
    /// <param name="hiddenDim">Hidden dimension of the encoder (must be positive)</param>
    /// <param name="decoderHiddenDim">Hidden dimension of the decoder; defaults to hiddenDim</param>
    /// <param name="activation">Activation used inside encoder/decoder; defaults to ReLU</param>
    /// <param name="beta">Weight of the KL term (β-VAE)</param>
    /// <param name="conditionDim">Dimension of an optional conditioning vector</param>
    public VAE(
        int inputDim,
        int latentDim,
        int hiddenDim,
        int? decoderHiddenDim = null,
        Func<ReverseGradTensor<T>, ReverseGradTensor<T>>? activation = null,
        float beta = 1.0f,
        int conditionDim = 0)
    {
        if (inputDim <= 0) throw new ArgumentOutOfRangeException(nameof(inputDim));
        if (latentDim <= 0) throw new ArgumentOutOfRangeException(nameof(latentDim));
        if (hiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenDim));
        if (conditionDim < 0) throw new ArgumentOutOfRangeException(nameof(conditionDim));

        this.inputDim = inputDim;
        this.latentDim = latentDim;
        this.hiddenDim = hiddenDim;
        this.conditionDim = conditionDim;

        var encoderInputDim = inputDim + conditionDim;
        var decoderInputDim = latentDim + conditionDim;

        encoderLayer1 = new Linear<T>(encoderInputDim, hiddenDim);
        encoderLayer2 = new Linear<T>(hiddenDim, hiddenDim);
        muHead = new Linear<T>(hiddenDim, latentDim);
        logVarHead = new Linear<T>(hiddenDim, latentDim);

        var decHidden = decoderHiddenDim ?? hiddenDim;
        decoderLayer1 = new Linear<T>(decoderInputDim, decHidden);
        decoderLayer2 = new Linear<T>(decHidden, inputDim);

        var betaData = new T[] { T.CreateChecked(beta) };
        this.beta = new Parameter<T>("Beta", betaData, requiresGrad: false);

        this.activation = activation ?? (x => Activation.Relu(x));

        RegisterModules(
            encoderLayer1, encoderLayer2,
            muHead, logVarHead,
            decoderLayer1, decoderLayer2);
        RegisterParameters(this.beta);
    }

    /// <summary>
    /// Runs a full encode → reparameterize → decode cycle without conditioning.
    /// </summary>
    /// <param name="x">The input tensor</param>
    /// <returns>The reconstructed output</returns>
    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> x)
    {
        var (mu, logVar) = Encode(x);
        var z = Reparameterize(mu, logVar);
        return Decode(z);
    }

    /// <summary>
    /// Runs a full encode → reparameterize → decode cycle with an optional conditioning vector.
    /// </summary>
    /// <param name="x">The input tensor</param>
    /// <param name="condition">Optional conditioning vector</param>
    /// <returns>The reconstructed output</returns>
    public ReverseGradTensor<T> Forward(ReverseGradTensor<T> x, ReverseGradTensor<T>? condition)
    {
        var (mu, logVar) = Encode(x, condition);
        var z = Reparameterize(mu, logVar);
        return Decode(z, condition);
    }

    ReverseGradTensor<T> IMultipleInputModule<T>.Forward(ReverseGradTensor<T> input1, ReverseGradTensor<T> input2)
        => Forward(input1, input2);

    /// <summary>
    /// Encodes an input into the latent mean and log-variance without conditioning.
    /// </summary>
    /// <param name="x">The input tensor</param>
    /// <returns>The latent mean and log-variance</returns>
    public (ReverseGradTensor<T> Mu, ReverseGradTensor<T> LogVar) Encode(ReverseGradTensor<T> x)
    {
        return Encode(x, condition: null);
    }

    /// <summary>
    /// Encodes an input into the latent mean and log-variance, optionally concatenating a
    /// conditioning vector along the feature axis.
    /// </summary>
    /// <param name="x">The input tensor</param>
    /// <param name="condition">Optional conditioning vector</param>
    /// <returns>The latent mean and log-variance</returns>
    public (ReverseGradTensor<T> Mu, ReverseGradTensor<T> LogVar) Encode(
        ReverseGradTensor<T> x,
        ReverseGradTensor<T>? condition = null)
    {
        var input = condition != null
            ? ReverseGradOperations.Concat(new[] { x, condition }, axis: 1)
            : x;

        var h = activation(encoderLayer1.Forward(input));
        h = activation(encoderLayer2.Forward(h));
        var mu = muHead.Forward(h);
        var logVar = logVarHead.Forward(h);
        return (mu, logVar);
    }

    /// <summary>
    /// Samples a latent using the reparameterization trick: <c>z = mu + exp(0.5·logVar) · ε</c>.
    /// Sampling is deterministic (uses the mean) when the module is in evaluation mode.
    /// </summary>
    /// <param name="mu">The latent mean</param>
    /// <param name="logVar">The latent log-variance</param>
    /// <param name="seed">Optional seed for reproducible sampling</param>
    /// <returns>The sampled latent</returns>
    public ReverseGradTensor<T> Reparameterize(ReverseGradTensor<T> mu, ReverseGradTensor<T> logVar, int? seed = null)
        => ModuleHelpers<T>.Reparameterize(mu, logVar, IsTraining, seed);

    /// <summary>
    /// Decodes a latent back into the input space without conditioning.
    /// </summary>
    /// <param name="z">The latent tensor</param>
    /// <returns>The reconstructed output</returns>
    public ReverseGradTensor<T> Decode(ReverseGradTensor<T> z)
    {
        return Decode(z, condition: null);
    }

    /// <summary>
    /// Decodes a latent back into the input space, optionally concatenating a conditioning
    /// vector along the feature axis.
    /// </summary>
    /// <param name="z">The latent tensor</param>
    /// <param name="condition">Optional conditioning vector</param>
    /// <returns>The reconstructed output</returns>
    public ReverseGradTensor<T> Decode(ReverseGradTensor<T> z, ReverseGradTensor<T>? condition = null)
    {
        var input = condition != null
            ? ReverseGradOperations.Concat(new[] { z, condition }, axis: 1)
            : z;

        var h = activation(decoderLayer1.Forward(input));
        return decoderLayer2.Forward(h);
    }

    /// <summary>
    /// Computes the ELBO loss as the sum of the squared-error reconstruction loss and the
    /// KL divergence, with the KL term weighted or left unweighted per <paramref name="lossType"/>.
    /// </summary>
    /// <param name="recon">The reconstructed output</param>
    /// <param name="original">The original input</param>
    /// <param name="mu">The latent mean</param>
    /// <param name="logVar">The latent log-variance</param>
    /// <param name="lossType">How to weight the KL term</param>
    /// <returns>The ELBO loss value</returns>
    public ReverseGradTensor<T> ElboLoss(
        ReverseGradTensor<T> recon,
        ReverseGradTensor<T> original,
        ReverseGradTensor<T> mu,
        ReverseGradTensor<T> logVar,
        ElboLossType lossType = ElboLossType.KldBeta)
    {
        var diff = ReverseGradOperations.Subtract(recon, original);
        var squared = ReverseGradOperations.Multiply(diff, diff);
        var reconLoss = ReverseGradOperations.Sum(squared);

        var kl = ReverseGradOperations.KlDivergence(mu, logVar);

        var betaResult = lossType switch
        {
            ElboLossType.KldBeta => ReverseGradOperations.Multiply(kl, beta.Tensor),
            ElboLossType.KldAnnealing => kl,
            _ => throw new ArgumentException($"Unknown ElboLossType: {lossType}", nameof(lossType))
        };

        return ReverseGradOperations.Add(reconLoss, betaResult);
    }
}
