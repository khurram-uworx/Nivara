using Microsoft.Extensions.AI;
using Nivara.AutoDiff.Nn;
using Nivara.Silk;
using NivaraChat.Shared;
using System.Text;

namespace NivaraChat.Transformer;

/// <summary>
/// An <see cref="IChatClient"/> backed by a <see cref="SilkTinyShakespeareSession"/> (Silk/OpenCL
/// GPU path for the batched TinyShakespeare transformer).
/// </summary>
internal sealed class SilkBatchedChatClient : CausalLmChatClientBase
{
    readonly SilkTinyShakespeareSession session;
    readonly TextTokenizer tokenizer;
    readonly int maxSeqLen;

    public SilkBatchedChatClient(
        SilkTinyShakespeareSession session,
        TextTokenizer tokenizer,
        int maxSeqLen,
        int maxNewTokens = 96,
        float? temperature = null,
        float? topP = null,
        int? seed = null,
        bool useKvCache = true)
        : base(temperature ?? 0f, Math.Clamp(topP ?? 1f, 0f, 1f), maxNewTokens, useKvCache, new Random(seed ?? 0))
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(tokenizer);

        this.session = session;
        this.tokenizer = tokenizer;
        this.maxSeqLen = maxSeqLen > 0 ? maxSeqLen : 1;
    }

    public override ChatClientMetadata? Metadata { get; } =
        new ChatClientMetadata("Nivara", null, "TinyShakespeare (Silk/OpenCL batched transformer)");

    public override void Dispose() => session.Dispose();

    protected override string RenderPrompt(IEnumerable<ChatMessage> messages)
        => FormatConversation(messages);

    protected override IReadOnlyList<int> Encode(string prompt)
        => tokenizer.Encode(prompt, addBosEos: true);

    protected override string DecodeToken(int tokenId)
        => tokenizer.Decode(new[] { tokenId });

    protected override string DecodeAll(IReadOnlyList<int> ids)
        => string.Concat(ids.Select(id => tokenizer.Decode(new[] { id })));

    protected override CausalLmOps BuildOps() => new(
        Prefill: w => session.ForwardPrefill(w),
        Cached: (t, p) => session.ForwardCached(t, p),
        LastLogits: w => session.ForwardPrefill(w),
        ResetCache: session.ResetCache,
        IsStopToken: id => id == tokenizer.EosToken,
        VocabSize: tokenizer.VocabSize,
        EosTokenId: tokenizer.EosToken,
        MaxSequenceLength: maxSeqLen);

    protected override bool WindowPrompt => true;
    protected override bool WindowNonCached => true;
    protected override int OverflowKeep => maxSeqLen - 1;

    static string FormatConversation(IEnumerable<ChatMessage> messages)
    {
        var sb = new StringBuilder();
        foreach (var message in messages)
        {
            if (string.IsNullOrEmpty(message.Text))
                continue;

            string role = message.Role == ChatRole.User ? "user" : "assistant";
            sb.Append(role).Append(": ").Append(message.Text).Append('\n');
        }
        sb.Append("assistant:");
        return sb.ToString();
    }
}
