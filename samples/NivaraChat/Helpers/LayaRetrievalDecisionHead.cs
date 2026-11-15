using System.Text.Json;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Utilities;
using Nivara.AI;
using NivaraChat.Modes;

namespace NivaraChat.Helpers;

/// <summary>
/// Evaluates a retrieved chunk against a local Laya decision checkpoint. When the checkpoint is not
/// present, the helper falls back to a retrieval-only decision so the sample still behaves.
/// </summary>
public sealed record LayaChunkDecision(
    string ChunkId,
    string Source,
    string Title,
    string Content,
    string Decision,
    double Confidence,
    double AnswerConfidence,
    double RetrievalScore,
    bool Accepted,
    bool Abstained,
    string Reason);

public static class LayaRetrievalDecisionHead
{
    public static (LayaQuestion Question, LayaSequence Sequence) BuildPrompt(
        Gpt2BpeTokenizer tokenizer,
        string query,
        string chunkId,
        string chunkContent,
        int maxLen,
        int headMaxLen)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(chunkId);
        ArgumentNullException.ThrowIfNull(chunkContent);

        var question = new LayaQuestion
        {
            Id = $"retrieval-{chunkId}",
            Type = LayaQuestionType.Choice,
            Instructions = $"Question: {query}\nIs this retrieved chunk relevant and answer-supporting?",
            Options =
            [
                new("relevant", "this chunk directly supports the answer or the user’s intent"),
                new("not_relevant", "this chunk is off-topic, stale, or not useful")
            ]
        };
        var stateIds = tokenizer.Encode(chunkContent);
        var sequence = LayaPromptBuilder.BuildSequence(
            tokenizer,
            chunkContent,
            question,
            maxLen,
            headMaxLen,
            stateIds: stateIds);
        return (question, sequence);
    }

    public static AdaptiveTrainingContext LoadTrainingContext(
        string modelDir,
        ResolvedLabels labels,
        IReadOnlyList<MultiViewChunk> chunks)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelDir);
        ArgumentNullException.ThrowIfNull(labels);
        ArgumentNullException.ThrowIfNull(chunks);
        if (labels.Kind != LabelKind.Judge || !RetrievalLabels.Accepts(labels))
            throw new ArgumentException("Adaptive head training requires accepted, resolved judge labels.", nameof(labels));

        var engine = new LayaDecisionEngine(modelDir, loadAdaptiveWeights: false);
        try
        {
            var examples = AdaptiveTrainingData.Build(labels, chunks, engine.ExtractFeatures);
            return new AdaptiveTrainingContext(engine.DetachHead(), examples);
        }
        finally
        {
            engine.Dispose();
        }
    }

    public static IReadOnlyList<LayaChunkDecision> Evaluate(
        string query,
        IReadOnlyList<RetrievedChunk> chunks,
        float confidenceThreshold)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(chunks);
        if (chunks.Count == 0)
            return [];

        var modelDir = Path.Combine(ModeHelpers.GetRepoRoot(), "samples", "data", "laya");
        if (!Directory.Exists(modelDir) || !File.Exists(Path.Combine(modelDir, "model.safetensors")))
        {
            return chunks
                .Select((chunk, index) => new LayaChunkDecision(
                    chunk.ChunkId,
                    chunk.Source,
                    chunk.Title,
                chunk.Content,
                    index == 0 ? "fallback" : "defer",
                    Math.Clamp(chunk.Score / 10.0, 0.0, 1.0),
                    Math.Clamp(chunk.Score / 10.0, 0.0, 1.0),
                    chunk.Score,
                    false,
                    false,
                    "Laya checkpoint unavailable; using retrieval rank as fallback."))
                .ToArray();
        }

        using var engine = new LayaDecisionEngine(modelDir);
        return chunks
            .Select(chunk =>
            {
                var decision = engine.Decide(query, chunk);
                bool accepted = decision.Choice == "relevant" && decision.AnswerConfidence >= confidenceThreshold;
                bool abstained = decision.Choice == "not_relevant" && decision.AnswerConfidence >= confidenceThreshold;
                string verdict = accepted ? "accept" : abstained ? "abstain" : "fallback";
                return new LayaChunkDecision(
                    chunk.ChunkId,
                    chunk.Source,
                    chunk.Title,
                    chunk.Content,
                    verdict,
                    decision.Confidence,
                    decision.AnswerConfidence,
                    chunk.Score,
                    accepted,
                    abstained,
                    accepted ? "Relevant chunk met the confidence threshold." :
                    abstained ? "Chunk rejected as off-topic or low-value." :
                    "Chunk did not meet the Laya confidence threshold.");
            })
            .ToArray();
    }

    public sealed class LayaDecisionEngine : IDisposable
    {
        readonly ModernBertEncoder<float, float> encoder;
        readonly LayaDecisionHead<float, float> head;
        readonly Gpt2BpeTokenizer tokenizer;
        readonly LayaCalibration calibration;
        readonly AgentConfig agent;
        bool ownsHead = true;
        bool disposed;

        public LayaDecisionEngine(
            string modelDir,
            bool loadAdaptiveWeights = true,
            string? adaptiveWeightsPath = null)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(modelDir);
            var config = LoadEncoderConfig(modelDir);
            agent = LoadAgentConfig(modelDir);
            var tensors = SafeTensorsLoader.Read<float>(Path.Combine(modelDir, "model.safetensors"));
            var loadedEncoder = ModernBertEncoder<float, float>.LoadWeights(tensors, config, prefix: "encoder");
            LayaDecisionHead<float, float>? loadedHead = null;
            try
            {
                loadedHead = LayaDecisionHead<float, float>.LoadWeights(tensors, agent.HeadLayers);
                if (loadAdaptiveWeights)
                {
                    AdaptiveHeadPersistence.LoadIfPresent(
                        loadedHead,
                        adaptiveWeightsPath ?? AdaptiveHeadPersistence.DefaultPath(ModeHelpers.GetRepoRoot()));
                }
                tokenizer = Gpt2BpeTokenizer.LoadFromTokenizerJson(Path.Combine(modelDir, "tokenizer", "tokenizer.json"), "[UNK]");
                calibration = new LayaCalibration(agent.Temperature, agent.TemperatureByOptions);
            }
            catch
            {
                loadedEncoder.Dispose();
                loadedHead?.Dispose();
                throw;
            }
            encoder = loadedEncoder;
            head = loadedHead;
        }

        public LayaDecisionHead<float, float> Head => head;

        public AdaptiveFrozenFeatures ExtractFeatures(string query, MultiViewChunk chunk)
        {
            if (GradientUtils.IsGradEnabled)
                throw new InvalidOperationException("ModernBERT feature extraction must run outside GradientUtils.Grad().");
            ArgumentNullException.ThrowIfNull(chunk);
            var (_, sequence) = BuildPrompt(
                tokenizer, query, chunk.ChunkId, chunk.Content, agent.MaxLen, agent.HeadMaxLen);
            var hidden = encoder.Forward(sequence.Ids, sequence.Length);
            if (!hidden.Data.TryGetSpan(out var span))
                throw new InvalidOperationException("ModernBERT encoder output is not available as a contiguous span.");
            return new AdaptiveFrozenFeatures(
                span.ToArray(), sequence.Length, hidden.Shape[1], sequence.Markers);
        }

        public LayaDecisionHead<float, float> DetachHead()
        {
            if (!ownsHead)
                throw new InvalidOperationException("The Laya head has already been detached.");
            ownsHead = false;
            return head;
        }

        public void Dispose()
        {
            if (disposed)
                return;
            disposed = true;
            encoder.Dispose();
            if (ownsHead)
                head.Dispose();
        }

        public LayaDecision Decide(string query, RetrievedChunk chunk)
        {
            var (question, sequence) = BuildPrompt(
                tokenizer,
                query,
                chunk.ChunkId,
                chunk.Content,
                agent.MaxLen,
                agent.HeadMaxLen);

            var hidden = encoder.Forward(sequence.Ids, sequence.Length);
            var output = head.Forward(hidden, question.Type, sequence.Markers, sequence.Length);
            var decision = calibration.Decode(question, output.Logits, output.ActionProbability);
            return decision;
        }

        static ModernBertConfig LoadEncoderConfig(string modelDir)
        {
            var path = Path.Combine(modelDir, "encoder", "config.json");
            if (!File.Exists(path))
                throw new FileNotFoundException($"Encoder config not found: {path}", path);
            return ModernBertConfig.FromJson(File.ReadAllText(path));
        }

        static AgentConfig LoadAgentConfig(string modelDir)
        {
            var path = Path.Combine(modelDir, "rl_agent_config.json");
            if (!File.Exists(path))
                throw new FileNotFoundException($"Agent config not found: {path}", path);

            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            var root = doc.RootElement;
            var temperature = root.GetProperty("temperature").EnumerateArray().Select(e => e.GetDouble()).ToArray();
            var byOptions = root.GetProperty("temperature_by_options").EnumerateObject()
                .ToDictionary(p => p.Name, p => p.Value.GetDouble());

            return new AgentConfig(
                root.GetProperty("max_len").GetInt32(),
                root.GetProperty("head_max_len").GetInt32(),
                root.GetProperty("head_layers").GetInt32(),
                temperature,
                byOptions);
        }
    }

    sealed record AgentConfig(
        int MaxLen,
        int HeadMaxLen,
        int HeadLayers,
        double[] Temperature,
        IReadOnlyDictionary<string, double> TemperatureByOptions);
}
