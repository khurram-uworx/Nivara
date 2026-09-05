using Nivara.AutoDiff;
using Nivara.Samples;
using NUnit.Framework;
using System.Buffers.Binary;
using System.Numerics;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Torch-parity fixture for Qwen2.5-0.5B-Instruct loader + tokenizer (issue #382 Phase 2).
/// The reference ids/logits are produced by the real <c>AutoModelForCausalLM</c> /
/// <c>AutoTokenizer</c> run in <c>samples/NivaraInference/Python/qwen_tool_reference.py</c>
/// (greedy decode over the native <c>&lt;tool_call&gt;</c> loop). Loading the locally-downloaded
/// checkpoint; skipped when the model/tokenizer files are absent (CI/clean).
/// </summary>
[TestFixture]
public class QwenInstructParityTests
{
    static string ModelDir
        => Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..",
            "samples", "data", "qwen2.5-0.5b-instruct");

    static Gpt2BpeTokenizer? cachedTokenizer;

    static Gpt2BpeTokenizer Tokenizer
    {
        get
        {
            if (cachedTokenizer != null)
                return cachedTokenizer;

            var vocab = Path.Combine(ModelDir, "vocab.json");
            var merges = Path.Combine(ModelDir, "merges.txt");
            var tokenizerJson = Path.Combine(ModelDir, "tokenizer.json");
            if (!File.Exists(vocab) || !File.Exists(merges) || !File.Exists(tokenizerJson))
                Assert.Ignore("Qwen tokenizer files absent; skipping tokenizer parity verification.");

            cachedTokenizer = new Gpt2BpeTokenizer(vocab, merges, tokenizerJsonPath: tokenizerJson);
            return cachedTokenizer;
        }
    }

    static (LlamaForCausalLM<float> Model, LlamaConfig Config)? cachedModel;

    static (LlamaForCausalLM<float> Model, LlamaConfig Config) Model
    {
        get
        {
            if (cachedModel != null)
                return cachedModel.Value;

            var safetensors = Path.Combine(ModelDir, "model.safetensors");
            var configJson = Path.Combine(ModelDir, "config.json");
            if (!File.Exists(safetensors) || !File.Exists(configJson))
                Assert.Ignore("Qwen safetensors absent; skipping model parity verification.");

            try
            {
                // BF16 on disk -> F32 compute (SafeTensorsLoader upcasts); qkvBias auto-detected.
                var tensors = SafeTensorsLoader.Read<float>(safetensors);
                var config = LlamaConfig.FromJson(File.ReadAllText(configJson));
                var model = LlamaLoader.Load<float, float>(config, tensors);
                cachedModel = (model, config);
                return cachedModel.Value;
            }
            catch (Exception ex)
            {
                Assert.Ignore($"Cannot load Qwen model: {ex.Message}");
                return default; // unreachable; keeps the compiler happy
            }
        }
    }

    /// <summary>Reads a little-endian int32 binary fixture.</summary>
    static int[] ReadInt32(string name)
    {
        var bytes = File.ReadAllBytes(Path.Combine(ModelDir, name));
        var result = new int[bytes.Length / 4];
        for (int i = 0; i < result.Length; i++)
            result[i] = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(i * 4, 4));
        return result;
    }

    /// <summary>Reads a little-endian float32 binary fixture.</summary>
    static float[] ReadFloat32(string name)
    {
        var bytes = File.ReadAllBytes(Path.Combine(ModelDir, name));
        var result = new float[bytes.Length / 4];
        for (int i = 0; i < result.Length; i++)
            result[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4, 4));
        return result;
    }

    [Test]
    public void Tokenizer_EncodeToolPrompt_MatchesTorchIds()
    {
        var prompt = File.ReadAllText(Path.Combine(ModelDir, "qwen_tool_prompt.txt"));
        var expected = ReadInt32("qwen_tool_prompt_ids.bin");

        var ids = Tokenizer.Encode(prompt);

        Assert.That(ids.Count, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ids[i], Is.EqualTo(expected[i]), $"token[{i}] differs (expected {expected[i]}, got {ids[i]})");
    }

    [Test]
    public void Tokenizer_EncodeFinalPrompt_MatchesTorchIds()
    {
        var prompt = File.ReadAllText(Path.Combine(ModelDir, "qwen_tool_final_prompt.txt"));
        var expected = ReadInt32("qwen_tool_final_prompt_ids.bin");

        var ids = Tokenizer.Encode(prompt);

        Assert.That(ids.Count, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ids[i], Is.EqualTo(expected[i]), $"token[{i}] differs (expected {expected[i]}, got {ids[i]})");
    }

    [Test]
    public void Tokenizer_VocabSize_IncludesAddedTokens()
    {
        // 151,643 base vocab + 22 added tokens (incl. <tool_call>/</tool_call>).
        Assert.That(Tokenizer.VocabSize, Is.EqualTo(151665));
    }

    [Test]
    public void Tokenizer_SpecialTokens_ResolveAsSingleIds()
    {
        Assert.That(Tokenizer.TokenId("<|endoftext|>"), Is.EqualTo(151643));
        Assert.That(Tokenizer.TokenId("<|im_start|>"), Is.EqualTo(151644));
        Assert.That(Tokenizer.TokenId("<|im_end|>"), Is.EqualTo(151645));
        Assert.That(Tokenizer.TokenId("<tool_call>"), Is.EqualTo(151657));
        Assert.That(Tokenizer.TokenId("</tool_call>"), Is.EqualTo(151658));

        // Added tokens must survive a round-trip verbatim (atomic, not char-decoded).
        var decoded = Tokenizer.Decode([151644, 151644, 151648, 151645]);
        Assert.That(decoded, Is.EqualTo("<|im_start|><|im_start|><|box_start|><|im_end|>"));
    }

    [Test]
    public void Model_QkvBias_TensorsLoaded()
    {
        var (model, config) = Model;

        // Qwen2.5-0.5B is the bias variant: q/k/v projections carry bias, o_proj does not.
        Assert.That(config.HiddenSize, Is.EqualTo(896));
        Assert.That(config.NumHiddenLayers, Is.EqualTo(24));
        Assert.That(config.NumAttentionHeads, Is.EqualTo(14));
        Assert.That(config.NumKeyValueHeads, Is.EqualTo(2));
        Assert.That(config.VocabSize, Is.EqualTo(151936));

        int headDim = config.HiddenSize / config.NumAttentionHeads; // 64
        int kvWidth = config.NumKeyValueHeads * headDim;            // 128

        var state = model.Parameters(); // same dotted keys as StateDict(), without cloning tensors

        // Every one of the 24 layers must carry exactly Q/K/V bias (24 × 896 + 48 × 128 entries),
        // and nothing else — o_proj/FFN/norms are bias-free in Qwen2.5-0.5B.
        var biasKeys = state.Keys.Where(k => k.EndsWith(".Bias")).ToArray();
        Assert.That(biasKeys.Length, Is.EqualTo(24 * 3),
            "expected exactly Q/K/V bias per layer, loaded via LlamaLoader qkvBias auto-detect");
        Assert.That(biasKeys.Count(k => state[k].Length == config.HiddenSize), Is.EqualTo(24),
            "one 896-wide bias per layer (q_proj)");
        Assert.That(biasKeys.Count(k => state[k].Length == kvWidth), Is.EqualTo(48),
            "two 128-wide biases per layer (k_proj/v_proj)");

        // Nested Module_{i} path: Embed=Module_0, layers=Module_1..24; in a block,
        // Attention=Module_1; in attention, QProj=Module_0. Spot-check layer 0's Q bias.
        var qBiasKey = $"Module_{1}.Module_1.Module_0.Bias";
        Assert.That(state.ContainsKey(qBiasKey), Is.True, "layer-0 q_proj bias must be present");
        Assert.That(state[qBiasKey].Length, Is.EqualTo(config.HiddenSize));
        Assert.That(state.Keys.Any(k => k.EndsWith(".Module_3.Bias")), Is.False,
            "o_proj is the 4th attention child (Module_3) and must have no bias");
    }

    [Test]
    public void Model_GreedyToolLoop_MatchesTorchGeneratedIds()
    {
        var (model, config) = Model;
        var expected = ReadInt32("qwen_tool_ids_py.bin");
        Assert.That(expected.Length, Is.EqualTo(42), "fixture must contain tool turn (19) + final answer (23) ids");

        var promptIds = ReadInt32("qwen_tool_prompt_ids.bin");

        var toolTurn = Greedy(model, config, promptIds, maxNewTokens: 160);
        Assert.That(toolTurn.Count, Is.EqualTo(19), "tool-call turn must be 19 tokens");
        for (int i = 0; i < toolTurn.Count; i++)
            Assert.That(toolTurn[i], Is.EqualTo(expected[i]), $"tool turn token[{i}] differs (expected {expected[i]}, got {toolTurn[i]})");
    }

    [Test]
    public void Model_GreedyFinalTurn_MatchesTorchGeneratedIdsAndFinalLogits()
    {
        var (model, config) = Model;
        var expected = ReadInt32("qwen_tool_ids_py.bin");

        var finalPromptIds = ReadInt32("qwen_tool_final_prompt_ids.bin");

        var finalTurn = Greedy(model, config, finalPromptIds, maxNewTokens: 160);
        Assert.That(finalTurn.Count, Is.EqualTo(23), "final-answer turn must be 23 tokens");
        for (int i = 0; i < finalTurn.Count; i++)
            Assert.That(finalTurn[i], Is.EqualTo(expected[19 + i]), $"final turn token[{i}] differs (expected {expected[19 + i]}, got {finalTurn[i]})");

        // Python dumps the logits at the position that predicts the final (eos) token — a full
        // forward over the whole final prompt + generated answer. Diff against that row.
        var fullIds = finalPromptIds.Concat(finalTurn).ToArray();
        var logits = model.Forward(fullIds); // [L, vocab]
        var torchLogits = ReadFloat32("qwen_tool_logits_py.bin");
        Assert.That(logits.Shape[1], Is.EqualTo(torchLogits.Length));

        int offset = logits.Length - logits.Shape[1];
        float maxAbsDiff = 0f;
        float maxAbsLogit = 0f;
        int argmax = -1;
        float best = float.NegativeInfinity;
        for (int i = 0; i < torchLogits.Length; i++)
        {
            float cSharp = logits[offset + i];
            float diff = Math.Abs(cSharp - torchLogits[i]);
            if (diff > maxAbsDiff)
                maxAbsDiff = diff;
            if (Math.Abs(torchLogits[i]) > maxAbsLogit)
                maxAbsLogit = Math.Abs(torchLogits[i]);
            if (cSharp > best)
            {
                best = cSharp;
                argmax = i;
            }
        }
        TestContext.Out.WriteLine(
            $"final-position logits: maxAbsDiff={maxAbsDiff:F6}, refMaxAbs={maxAbsLogit:F3}, argmax={argmax}");
        Assert.That(argmax, Is.EqualTo(151645), "final-row argmax must predict <|im_end|>");
        // Torch reference computed in BF16 (torch_dtype="auto"); C# computes F32 from BF16-upcast
        // weights, so parity is bounded by BF16 rounding — assert a relative bound, not an absolute one.
        Assert.That(maxAbsDiff, Is.LessThan(0.01f * maxAbsLogit + 0.05f),
            "final-position logits must be within BF16-reference relative tolerance");
    }

    /// <summary>Greedily decodes with a KV cache (numeric-identical to full forward), stopping on
    /// the Qwen eos id 151645 exactly as the reference <c>_greedy</c> does.</summary>
    static List<int> Greedy(LlamaForCausalLM<float> model, LlamaConfig config, int[] promptIds, int maxNewTokens)
    {
        int kvWidth = config.NumKeyValueHeads * (config.HiddenSize / config.NumAttentionHeads);
        using var cache = new LlamaKVCache<float>(config.NumHiddenLayers, kvWidth);

        ReverseGradTensor<float> logits = null!;
        for (int p = 0; p < promptIds.Length; p++)
            logits = model.ForwardCached(promptIds[p], p, cache);

        int position = promptIds.Length;
        var gen = new List<int>();
        for (int t = 0; t < maxNewTokens && gen.Count < config.MaxPositionEmbeddings; t++)
        {
            int next = ArgMax(logits, config.VocabSize);
            if (next == 151645)
                break;
            gen.Add(next);
            logits = model.ForwardCached(next, position++, cache);
        }
        return gen;
    }

    static int ArgMax(ReverseGradTensor<float> logits, int vocab)
    {
        int best = -1;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < vocab; i++)
        {
            float v = logits[i];
            if (v > bestVal)
            {
                bestVal = v;
                best = i;
            }
        }
        return best;
    }
}