namespace NivaraChat.Shared;

/// <summary>
/// A compositional bundle of the operations a causal-LM session exposes to the shared
/// generation loop. Clients build this from their own model/session instead of
/// implementing a common interface.
/// </summary>
internal readonly record struct CausalLmOps(
    Func<int[], float[]> Prefill,
    Func<int, int, float[]> Cached,
    Func<int[], float[]> LastLogits,
    Action ResetCache,
    Func<int, bool> IsStopToken,
    int VocabSize,
    int EosTokenId,
    int MaxSequenceLength);
