using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Fully convolutional variational autoencoder. Encodes an image with a stack of strided
/// convolutions into per-channel latent mean and log-variance, samples via the
/// reparameterization trick, and decodes with a stack of transposed convolutions.
/// </summary>
public sealed class ConvVAE<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly Conv2d<T>[] encoderConvs;
    readonly Conv2d<T> muConv;
    readonly Conv2d<T> logVarConv;
    readonly ConvTranspose2d<T>[] decoderConvs;
    readonly ConvTranspose2d<T> reconConv;
    readonly int latentChannels;
    readonly int spatialSize;

    /// <summary>Gets the number of latent channels (feature-map channels of the latent).</summary>
    public int LatentChannels => latentChannels;
    /// <summary>Gets the spatial size of the latent feature maps.</summary>
    public int SpatialSize => spatialSize;

    /// <summary>
    /// Creates a convolutional variational autoencoder.
    /// </summary>
    /// <param name="inputChannels">Number of channels in the input image</param>
    /// <param name="encoderChannels">Channel counts of the encoder convolutions (at least one entry)</param>
    /// <param name="latentChannels">Number of latent channels (must be positive)</param>
    /// <param name="spatialSize">Spatial size of the latent feature maps (must be positive)</param>
    /// <param name="kernelSize">Convolution kernel size</param>
    /// <param name="stride">Convolution stride</param>
    /// <param name="padding">Convolution padding</param>
    public ConvVAE(
        int inputChannels,
        int[] encoderChannels,
        int latentChannels,
        int spatialSize,
        int kernelSize = 3,
        int stride = 2,
        int padding = 1)
    {
        if (encoderChannels == null || encoderChannels.Length == 0)
            throw new ArgumentException("Encoder channels must have at least one entry.", nameof(encoderChannels));
        if (latentChannels <= 0) throw new ArgumentOutOfRangeException(nameof(latentChannels));
        if (spatialSize <= 0) throw new ArgumentOutOfRangeException(nameof(spatialSize));

        this.latentChannels = latentChannels;
        this.spatialSize = spatialSize;

        encoderConvs = new Conv2d<T>[encoderChannels.Length];
        for (int i = 0; i < encoderChannels.Length; i++)
        {
            int inC = i == 0 ? inputChannels : encoderChannels[i - 1];
            encoderConvs[i] = new Conv2d<T>(inC, encoderChannels[i], kernelSize, stride, padding, bias: false);
        }

        int lastEncCh = encoderChannels[^1];
        muConv = new Conv2d<T>(lastEncCh, latentChannels, 1, bias: false);
        logVarConv = new Conv2d<T>(lastEncCh, latentChannels, 1, bias: false);

        int decoderSteps = encoderChannels.Length;
        var decoderChannels = new int[decoderSteps];
        for (int i = 0; i < decoderSteps; i++)
            decoderChannels[i] = encoderChannels[decoderSteps - 1 - i];

        decoderConvs = new ConvTranspose2d<T>[decoderSteps];
        for (int i = 0; i < decoderSteps; i++)
        {
            int inC = i == 0 ? latentChannels : decoderChannels[i - 1];
            decoderConvs[i] = new ConvTranspose2d<T>(inC, decoderChannels[i], kernelSize, stride, padding, bias: false);
        }

        reconConv = new ConvTranspose2d<T>(decoderChannels[^1], inputChannels, 1, bias: true);

        RegisterModules(encoderConvs);
        RegisterModules(muConv, logVarConv);
        RegisterModules(decoderConvs);
        RegisterModules(reconConv);
    }

    /// <summary>
    /// Runs a full encode → reparameterize → decode cycle.
    /// </summary>
    /// <param name="input">The input image tensor (rank 4)</param>
    /// <returns>The reconstructed output</returns>
    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        var (mu, logVar) = Encode(input);
        var z = Reparameterize(mu, logVar);
        return Decode(z);
    }

    /// <summary>
    /// Encodes an image into latent mean and log-variance feature maps via the encoder
    /// convolution stack (each followed by ReLU).
    /// </summary>
    /// <param name="input">The input image tensor (rank 4)</param>
    /// <returns>The latent mean and log-variance feature maps</returns>
    public (ReverseGradTensor<T> Mu, ReverseGradTensor<T> LogVar) Encode(ReverseGradTensor<T> input)
    {
        var h = input;
        for (int i = 0; i < encoderConvs.Length; i++)
        {
            h = encoderConvs[i].Forward(h);
            h = Activation.Relu(h);
        }

        var mu = muConv.Forward(h);
        var logVar = logVarConv.Forward(h);
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
    /// Decodes a latent back into an image via the decoder transposed-convolution stack
    /// (each followed by ReLU) and the final 1×1 reconstruction convolution.
    /// </summary>
    /// <param name="z">The latent tensor</param>
    /// <returns>The reconstructed image</returns>
    public ReverseGradTensor<T> Decode(ReverseGradTensor<T> z)
    {
        var h = z;
        for (int i = 0; i < decoderConvs.Length; i++)
        {
            h = decoderConvs[i].Forward(h);
            h = Activation.Relu(h);
        }

        return reconConv.Forward(h);
    }

    /// <summary>
    /// Computes the ELBO loss as the sum of the squared-error reconstruction loss and the
    /// (unweighted) KL divergence.
    /// </summary>
    /// <param name="recon">The reconstructed output</param>
    /// <param name="original">The original input</param>
    /// <param name="mu">The latent mean</param>
    /// <param name="logVar">The latent log-variance</param>
    /// <returns>The ELBO loss value</returns>
    public ReverseGradTensor<T> ElboLoss(
        ReverseGradTensor<T> recon,
        ReverseGradTensor<T> original,
        ReverseGradTensor<T> mu,
        ReverseGradTensor<T> logVar)
    {
        var diff = ReverseGradOperations.Subtract(recon, original);
        var squared = ReverseGradOperations.Multiply(diff, diff);
        var reconLoss = ReverseGradOperations.Sum(squared);

        var kl = ReverseGradOperations.KlDivergence(mu, logVar);

        return ReverseGradOperations.Add(reconLoss, kl);
    }
}
