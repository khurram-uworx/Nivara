using System.Text.Json;
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

        var engine = new LayaDecisionEngine(modelDir);
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

    sealed class LayaDecisionEngine
    {
        readonly ModernBertEncoder<float, float> encoder;
        readonly LayaDecisionHead<float, float> head;
        readonly Gpt2BpeTokenizer tokenizer;
        readonly LayaCalibration calibration;
        readonly AgentConfig agent;

        public LayaDecisionEngine(string modelDir)
        {
            var config = LoadEncoderConfig(modelDir);
            agent = LoadAgentConfig(modelDir);
            var tensors = SafeTensorsLoader.Read<float>(Path.Combine(modelDir, "model.safetensors"));
            encoder = ModernBertEncoder<float, float>.LoadWeights(tensors, config, prefix: "encoder");
            head = LayaDecisionHead<float, float>.LoadWeights(tensors, agent.HeadLayers);
            tokenizer = Gpt2BpeTokenizer.LoadFromTokenizerJson(Path.Combine(modelDir, "tokenizer", "tokenizer.json"), "[UNK]");
            calibration = new LayaCalibration(agent.Temperature, agent.TemperatureByOptions);
        }

        public LayaDecision Decide(string query, RetrievedChunk chunk)
        {
            var question = new LayaQuestion
            {
                Id = $"retrieval-{chunk.ChunkId}",
                Type = LayaQuestionType.Choice,
                Instructions = $"Question: {query}\nIs this retrieved chunk relevant and answer-supporting?",
                Options =
                [
                    new("relevant", "this chunk directly supports the answer or the user’s intent"),
                    new("not_relevant", "this chunk is off-topic, stale, or not useful")
                ]
            };

            var stateIds = tokenizer.Encode(query);
            var sequence = LayaPromptBuilder.BuildSequence(
                tokenizer,
                query,
                question,
                agent.MaxLen,
                agent.HeadMaxLen,
                stateIds: stateIds);

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
