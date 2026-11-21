using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Nivara.AI;
using Nivara.Samples;

namespace NivaraFineTuning;

public sealed record WorklogSftExample(
    [property: JsonPropertyName("messages")] List<WorklogMessage> Messages);

public sealed record WorklogMessage(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] string Content);

public sealed record WorklogTokenizedExample(
    int[] InputIds,
    int[] Labels,
    int PromptTokenCount,
    int AnswerTokenCount);

public sealed record WorklogSftSplit(
    List<WorklogSftExample> Train,
    List<WorklogSftExample> Validation,
    List<WorklogSftExample> Test,
    string ProvenanceHash);

public static class WorklogSftDataset
{
    const int IgnoreIndex = -100;

    public static List<WorklogSftExample> LoadJsonl(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Worklog SFT JSONL not found at {path}", path);

        var examples = new List<WorklogSftExample>();
        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var example = JsonSerializer.Deserialize<WorklogSftExample>(line)
                ?? throw new InvalidOperationException($"Failed to deserialize worklog example in '{path}'.");
            if (example.Messages is null || example.Messages.Count == 0)
                throw new InvalidOperationException($"Worklog example in '{path}' has no messages.");
            examples.Add(example);
        }
        return examples;
    }

    public static void SaveJsonl(string path, IReadOnlyList<WorklogSftExample> examples, bool force = false)
    {
        if (File.Exists(path) && !force)
            throw new InvalidOperationException($"Output file '{path}' already exists. Use --force to overwrite.");
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

        using var writer = new StreamWriter(path, false, new UTF8Encoding(false));
        foreach (var ex in examples)
            writer.WriteLine(JsonSerializer.Serialize(ex));
    }

    public static WorklogSftSplit SplitDeterministic(
        IReadOnlyList<WorklogSftExample> examples,
        int seed = 42,
        double trainRatio = 0.8,
        double valRatio = 0.1)
    {
        if (examples.Count == 0) throw new ArgumentException("No examples to split.", nameof(examples));
        if (trainRatio <= 0 || valRatio < 0 || trainRatio + valRatio >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(trainRatio), "Invalid split ratios.");

        var indices = Enumerable.Range(0, examples.Count).ToArray();
        var rng = new Random(seed);
        for (int i = indices.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        int trainCount = Math.Max(1, (int)Math.Floor(examples.Count * trainRatio));
        int valCount = Math.Max(1, (int)Math.Floor(examples.Count * valRatio));
        if (trainCount + valCount >= examples.Count) valCount = Math.Max(0, examples.Count - trainCount - 1);

        var train = new List<WorklogSftExample>(trainCount);
        var val = new List<WorklogSftExample>(valCount);
        var test = new List<WorklogSftExample>(examples.Count - trainCount - valCount);

        for (int i = 0; i < indices.Length; i++)
        {
            var ex = examples[indices[i]];
            if (i < trainCount) train.Add(ex);
            else if (i < trainCount + valCount) val.Add(ex);
            else test.Add(ex);
        }

        return new WorklogSftSplit(train, val, test, ComputeProvenanceHash(examples, seed, trainRatio, valRatio));
    }

    public static string ComputeProvenanceHash(IReadOnlyList<WorklogSftExample> examples, int seed, double trainRatio, double valRatio)
    {
        using var sha = SHA256.Create();
        var sb = new StringBuilder();
        sb.Append(seed).Append('|').Append(trainRatio).Append('|').Append(valRatio).Append('|').Append(examples.Count).Append('\n');
        foreach (var ex in examples)
        {
            foreach (var m in ex.Messages)
                sb.Append(m.Role).Append('\0').Append(m.Content).Append('\0');
            sb.Append('\n');
        }
        var bytes = sha.ComputeHash(Encoding.UTF8.GetBytes(sb.ToString()));
        return Convert.ToHexString(bytes).ToLowerInvariant();
    }

    public static List<WorklogTokenizedExample> Tokenize(
        IReadOnlyList<WorklogSftExample> examples,
        Gpt2BpeTokenizer tokenizer,
        string template,
        int maxLen)
    {
        ArgumentNullException.ThrowIfNull(examples);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentException.ThrowIfNullOrWhiteSpace(template);
        if (maxLen <= 2) throw new ArgumentOutOfRangeException(nameof(maxLen));

        var result = new List<WorklogTokenizedExample>(examples.Count);
        foreach (var ex in examples)
        {
            var tokenized = TokenizeOne(ex, tokenizer, template, maxLen);
            if (tokenized is not null)
                result.Add(tokenized);
        }
        return result;
    }

    public static WorklogTokenizedExample? TokenizeOne(
        WorklogSftExample example,
        Gpt2BpeTokenizer tokenizer,
        string template,
        int maxLen)
    {
        ArgumentNullException.ThrowIfNull(example);
        ArgumentNullException.ThrowIfNull(tokenizer);

        var messages = example.Messages ?? throw new InvalidOperationException("Example has no messages.");
        int lastAssistant = -1;
        for (int i = messages.Count - 1; i >= 0; i--)
        {
            if (string.Equals(messages[i].Role, "assistant", StringComparison.OrdinalIgnoreCase))
            {
                lastAssistant = i;
                break;
            }
        }
        if (lastAssistant < 0)
            throw new InvalidOperationException("Worklog SFT example must contain an assistant message.");

        var promptMessages = messages.Take(lastAssistant).ToList();
        var answer = messages[lastAssistant].Content ?? string.Empty;

        var promptText = RenderMessages(promptMessages, template, addGenerationPrompt: true);
        var fullText = RenderMessages(messages.Take(lastAssistant + 1).ToList(), template, addGenerationPrompt: false);

        var promptIds = tokenizer.Encode(promptText);
        var fullIds = tokenizer.Encode(fullText);

        if (fullIds.Count > maxLen)
        {
            // Keep answer region by truncating prompt from the left when possible.
            int answerLen = Math.Max(0, fullIds.Count - promptIds.Count);
            int roomForPrompt = Math.Max(0, maxLen - answerLen);
            if (roomForPrompt <= 0)
                return null;

            var promptList = promptIds.ToList();
            if (promptList.Count > roomForPrompt)
                promptList.RemoveRange(0, promptList.Count - roomForPrompt);

            var fullList = new List<int>(promptList.Count + answerLen);
            fullList.AddRange(promptList);
            fullList.AddRange(fullIds.Skip(promptIds.Count));
            if (fullList.Count > maxLen)
                fullList.RemoveRange(0, fullList.Count - maxLen);

            fullIds = fullList;
            promptIds = promptList;
        }

        int promptCount = Math.Min(promptIds.Count, fullIds.Count);
        // If prompt encoding is not a prefix due tokenizer boundary effects, fall back to best-effort boundary.
        int prefix = 0;
        while (prefix < promptCount && fullIds[prefix] == promptIds[prefix]) prefix++;
        promptCount = prefix == promptCount ? promptCount : prefix;

        int answerCount = fullIds.Count - promptCount;
        if (answerCount <= 0) return null;

        var inputIds = fullIds.ToArray();
        var labels = new int[inputIds.Length];
        for (int i = 0; i < labels.Length; i++)
            labels[i] = i < promptCount ? IgnoreIndex : inputIds[i];

        return new WorklogTokenizedExample(inputIds, labels, promptCount, answerCount);
    }

    static string RenderMessages(IReadOnlyList<WorklogMessage> messages, string template, bool addGenerationPrompt)
    {
        var chatMessages = messages.Select(m => new Microsoft.Extensions.AI.ChatMessage(
            m.Role.ToLowerInvariant() switch
            {
                "system" => Microsoft.Extensions.AI.ChatRole.System,
                "user" => Microsoft.Extensions.AI.ChatRole.User,
                "assistant" => Microsoft.Extensions.AI.ChatRole.Assistant,
                "tool" => Microsoft.Extensions.AI.ChatRole.Tool,
                _ => Microsoft.Extensions.AI.ChatRole.User
            },
            m.Content ?? string.Empty)).ToList();

        return template.ToLowerInvariant() switch
        {
            "qwen" => QwenChatTemplate.Render(chatMessages, addGenerationPrompt),
            "smollm" => SmollmChatTemplate.Render(chatMessages, addGenerationPrompt),
            _ => throw new ArgumentException($"Unknown template '{template}'. Expected qwen or smollm.", nameof(template))
        };
    }
}
