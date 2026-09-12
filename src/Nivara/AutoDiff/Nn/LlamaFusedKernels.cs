namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Runtime toggles for the inference-only fused Llama kernels. Mirroring
/// <see cref="Nivara.Primitives.NivaraPrimitives.UseWidenSimd"/>, the default is on so cached
/// inference outside a <see cref="Utilities.GradientUtils.Grad"/> scope routes through the fused
/// per-token decoder-block path; parity tests flip it to <c>false</c> to force the
/// byte-compatible per-op chain for A/B comparisons. Grad-enabled training is unaffected (the
/// fused kernels never run inside a Grad scope).
/// </summary>
public static class LlamaFusedKernels
{
    /// <summary>
    /// Gets or sets whether cached inference (decode and prefill) routes through the fused
    /// single-token decoder-block kernel instead of the per-op chain.
    /// </summary>
    public static bool DecoderBlockFused { get; set; } = true;
}
