using Nivara.AI;
using Nivara.Silk;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using TypeSafeAI;

namespace NivaraInference;

/// <summary>
/// Pure, dependency-free mapping between the TypeSafe SDK surface and the Laya question/decision types.
/// Keeping this separate from the device-owning client is what lets the serializer and mapping be unit
/// tested without a checkpoint or an OpenCL device.
/// </summary>
public static class NivaraTypeSafeLayaMapping
{
    public static string ContentToText(TypeSafeContent? content)
    {
        if (!content.HasValue || content.Value.IsEmpty)
            return string.Empty;
        if (content.Value.IsText)
            return content.Value.Text ?? string.Empty;
        if (content.Value.IsJson && content.Value.Node is not null)
            return SerializeNode(content.Value.Node);
        return content.Value.ToString() ?? string.Empty;
    }

    public static LayaQuestion ToLayaQuestion(string id, Question question)
    {
        if (question is NoulQuestion nqdbg)
            ArgumentNullException.ThrowIfNull(question);

        return question switch
        {
            ChoiceQuestion cq => new LayaQuestion
            {
                Id = id,
                Type = LayaQuestionType.Choice,
                Instructions = ContentToText(cq.Instructions),
                Options = MapChoiceOptions(cq.Criteria),
            },
            ScoreQuestion sq => new LayaQuestion
            {
                Id = id,
                Type = LayaQuestionType.Score,
                Instructions = ContentToText(sq.Instructions),
                ScoreCriteria = MapScoreCriteria(sq.Criteria),
            },
            NoulQuestion nq => new LayaQuestion
            {
                Id = id,
                Type = LayaQuestionType.Noul,
                Instructions = ContentToText(nq.Instructions),
                NoulFalseCriterion = nq.Criteria is null ? string.Empty : ContentToText(nq.Criteria!.No),
                NoulTrueCriterion = nq.Criteria is null ? string.Empty : ContentToText(nq.Criteria!.Yes),
            },
            _ => throw new ArgumentException(
                $"Unsupported question type '{question!.GetType().Name}'", nameof(question)),
        };
    }

    public static Answer ToAnswer(Question question, LayaDecision decision)
    {
        ArgumentNullException.ThrowIfNull(question);
        ArgumentNullException.ThrowIfNull(decision);

        switch (question)
        {
            case ChoiceQuestion cq:
                {
                    var probabilities = new Dictionary<string, double>(StringComparer.Ordinal);
                    int i = 0;
                    foreach (var (label, _) in cq.Criteria)
                    {
                        probabilities[label] = i < decision.Probabilities.Count
                            ? decision.Probabilities[i]
                            : 0.0;
                        i++;
                    }
                    var answer = new ChoiceAnswer
                    {
                        Choice = decision.Choice ?? string.Empty,
                        Probabilities = probabilities,
                        Confidence = decision.AnswerConfidence,
                        AdditionalData = new Dictionary<string, JsonElement>(),
                    };
                    FillAdditionalData(answer.AdditionalData, decision);
                    return answer;
                }

            case ScoreQuestion sq:
                {
                    var probabilities = new Dictionary<int, double>();
                    for (int i = 0; i < decision.Probabilities.Count; i++)
                        probabilities[i] = decision.Probabilities[i];

                    var legend = new Dictionary<int, TypeSafeContent?>();
                    for (int i = 0; i < sq.Criteria.Count; i++)
                        legend[i] = sq.Criteria[i];

                    var answer = new ScoreAnswer
                    {
                        Score = decision.Score ?? 0.0,
                        Legend = legend,
                        Probabilities = probabilities,
                        Confidence = decision.AnswerConfidence,
                        AdditionalData = new Dictionary<string, JsonElement>(),
                    };
                    FillAdditionalData(answer.AdditionalData, decision);
                    return answer;
                }

            case NoulQuestion:
                {
                    var answer = new NoulAnswer
                    {
                        Probability = decision.NoulProbability ?? 0.0,
                        AdditionalData = new Dictionary<string, JsonElement>(),
                    };
                    FillAdditionalData(answer.AdditionalData, decision);
                    return answer;
                }

            default:
                throw new ArgumentException(
                    $"Unsupported question type '{question.GetType().Name}'", nameof(question));
        }
    }

    private static void FillAdditionalData(IDictionary<string, JsonElement> data, LayaDecision decision)
    {
        data["answer_confidence"] = JsonSerializer.SerializeToElement(decision.AnswerConfidence);
        data["confidence"] = JsonSerializer.SerializeToElement(decision.Confidence);
        data["temperature"] = JsonSerializer.SerializeToElement(decision.Temperature.Scale);
        data["act_probability"] = JsonSerializer.SerializeToElement(decision.ActionProbability);
    }

    private static IReadOnlyList<LayaChoiceOption> MapChoiceOptions(
        IReadOnlyDictionary<string, TypeSafeContent?> criteria)
    {
        var options = new List<LayaChoiceOption>(criteria.Count);
        foreach (var (label, content) in criteria)
        {
            var description = content is null ? null : ContentToText(content);
            if (description is not null && description.Length == 0)
                description = null;
            options.Add(new LayaChoiceOption(label, description));
        }
        return options;
    }

    private static IReadOnlyList<string?> MapScoreCriteria(
        IReadOnlyList<TypeSafeContent?> criteria)
    {
        var result = new List<string?>(criteria.Count);
        foreach (var content in criteria)
            result.Add(content is null ? null : ContentToText(content));
        return result;
    }

    private static string SerializeNode(JsonNode node)
    {
        var sb = new StringBuilder();
        SerializeNode(node, sb);
        return sb.ToString();
    }

    private static void SerializeNode(JsonNode? node, StringBuilder sb)
    {
        if (node is null)
        {
            sb.Append("null");
            return;
        }

        switch (node)
        {
            case JsonObject obj:
                sb.Append('{');
                bool first = true;
                foreach (var (name, value) in obj)
                {
                    if (!first) sb.Append(", ");
                    first = false;
                    sb.Append(JsonSerializer.Serialize(name));
                    sb.Append(": ");
                    SerializeNode(value, sb);
                }
                sb.Append('}');
                return;

            case JsonArray arr:
                sb.Append('[');
                for (int i = 0; i < arr.Count; i++)
                {
                    if (i > 0) sb.Append(", ");
                    SerializeNode(arr[i], sb);
                }
                sb.Append(']');
                return;

            case JsonValue value:
                {
                    try
                    {
                        var element = value.GetValue<JsonElement>();
                        switch (element.ValueKind)
                        {
                            case JsonValueKind.String:
                                // Preserve the original token text, raw non-ASCII stays raw.
                                sb.Append(element.GetRawText());
                                return;
                            case JsonValueKind.Number:
                            case JsonValueKind.True:
                            case JsonValueKind.False:
                            case JsonValueKind.Null:
                                sb.Append(element.GetRawText());
                                return;
                            default:
                                sb.Append(value.ToJsonString());
                                return;
                        }
                    }
                    catch
                    {
                        sb.Append(value.ToJsonString());
                        return;
                    }
                }

            default:
                sb.Append(node.ToJsonString());
                return;
        }
    }
}

public sealed record SystemOnePhaseTimings(
    long StateTextMs,
    long StateTokenizeMs,
    long QuestionMappingMs,
    long PromptBuildMs,
    long ForwardMs,
    long EncodeMs,
    long HeadMs,
    long DecodeMs,
    long AnswerMappingMs,
    long AllocationsMs,
    long TotalMs);

public sealed class NivaraTypeSafeClient : ITypeSafeClient, IDisposable
{
    readonly string modelDir;
    readonly object gate = new();
    readonly Laya.AgentConfig agent;
    readonly ModernBertConfig encoderConfig;
    readonly Gpt2BpeTokenizer tokenizer;
    readonly LayaCalibration calibration;
    readonly ILayaGpuSession session;
    readonly Stopwatch loadSw = Stopwatch.StartNew();
    bool disposed;

    public SystemOnePhaseTimings? LastPhaseTimings { get; private set; }

    public NivaraTypeSafeClient(string modelDir)
    {
        this.modelDir = modelDir;
        string path = Path.Combine(modelDir, "model.safetensors");
        if (!File.Exists(path))
            throw new FileNotFoundException($"Model file not found: {path}", path);

        agent = Laya.LoadAgentConfig(modelDir);
        encoderConfig = Laya.LoadEncoderConfig(modelDir);
        tokenizer = Laya.LoadTokenizer(modelDir);
        calibration = new LayaCalibration(agent.Temperature, agent.TemperatureByOptions);

        var tensors = SafeTensorsLoader.Read(path);
        session = SilkLayaSession.Open(encoderConfig, tensors, agent.HeadLayers);
        loadSw.Stop();
    }

    public string DeviceName => session.DeviceName;
    public long LoadMilliseconds => loadSw.ElapsedMilliseconds;

    public IModelsResource Models { get; } = new ModelsResource();

    public Task<SystemOneResponse> SystemOneAsync(
        SystemOneRequest request, RequestOptions? options = null, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var totalSw = Stopwatch.StartNew();

        var sw = Stopwatch.StartNew();
        var stateText = NivaraTypeSafeLayaMapping.ContentToText(request.State);
        var stateTextMs = sw.ElapsedMilliseconds;
        sw.Restart();
        var stateIds = tokenizer.Encode(stateText.Replace("[MASK]", " ", StringComparison.Ordinal));
        var stateTokenizeMs = sw.ElapsedMilliseconds;

        sw.Restart();
        var layaQuestions = new Dictionary<string, LayaQuestion>(request.Questions.Count, StringComparer.Ordinal);
        foreach (var (id, question) in request.Questions)
            layaQuestions[id] = NivaraTypeSafeLayaMapping.ToLayaQuestion(id, question!);
        var questionMappingMs = sw.ElapsedMilliseconds;

        sw.Restart();
        var sequences = new Dictionary<string, LayaSequence>(request.Questions.Count, StringComparer.Ordinal);
        foreach (var (id, _) in request.Questions)
            sequences[id] = LayaPromptBuilder.BuildSequence(
                tokenizer, stateText, layaQuestions[id], agent.MaxLen, agent.HeadMaxLen, stateIds: stateIds);
        var promptBuildMs = sw.ElapsedMilliseconds;

        var answers = new Dictionary<string, Answer>(request.Questions.Count, StringComparer.Ordinal);
        int inputTokens = 0;
        long forwardMs = 0;
        long encodeMs = 0;
        long headMs = 0;
        long decodeMs = 0;
        long answerMappingMs = 0;

        lock (gate)
        {
            foreach (var (id, question) in request.Questions)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var layaQuestion = layaQuestions[id];
                var sequence = sequences[id];

                var fwdSw = Stopwatch.StartNew();
                var output = session.Forward(sequence.Ids, sequence.Length, layaQuestion.Type, sequence.Markers);
                forwardMs += fwdSw.ElapsedMilliseconds;
                encodeMs += session.LastEncodeMilliseconds;
                headMs += session.LastHeadMilliseconds;

                var decSw = Stopwatch.StartNew();
                var decision = calibration.Decode(layaQuestion, output.Logits, output.ActionProbability);
                decodeMs += decSw.ElapsedMilliseconds;

                var mapSw = Stopwatch.StartNew();
                answers[id] = NivaraTypeSafeLayaMapping.ToAnswer(question!, decision);
                answerMappingMs += mapSw.ElapsedMilliseconds;

                inputTokens += sequence.Length;
            }
        }

        var allocSw = Stopwatch.StartNew();
        var nouls = answers.Where(kv => kv.Value is NoulAnswer)
            .ToDictionary(kv => kv.Key, kv => (NoulAnswer)kv.Value);
        var choices = answers.Where(kv => kv.Value is ChoiceAnswer)
            .ToDictionary(kv => kv.Key, kv => (ChoiceAnswer)kv.Value);
        var scores = answers.Where(kv => kv.Value is ScoreAnswer)
            .ToDictionary(kv => kv.Key, kv => (ScoreAnswer)kv.Value);

        var response = new SystemOneResponse
        {
            Model = "convaiinnovations/laya",
            Answers = answers,
            Usage = new Usage { InputTokens = inputTokens, OutputTokens = 0 },
        };
        var allocationsMs = allocSw.ElapsedMilliseconds;
        totalSw.Stop();

        LastPhaseTimings = new SystemOnePhaseTimings(
            stateTextMs,
            stateTokenizeMs,
            questionMappingMs,
            promptBuildMs,
            forwardMs,
            encodeMs,
            headMs,
            decodeMs,
            answerMappingMs,
            allocationsMs,
            totalSw.ElapsedMilliseconds);

        return Task.FromResult(response);
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        session.Dispose();
    }

    sealed class ModelsResource : IModelsResource
    {
        public Task<ModelsResponse> ListAsync(RequestOptions? options = null, CancellationToken cancellationToken = default)
            => throw new NotSupportedException("Model listing is not supported by the in-process engine.");
    }
}
