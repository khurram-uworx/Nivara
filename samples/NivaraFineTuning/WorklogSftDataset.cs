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

    public static List<WorklogSftExample> GenerateSynthetic(int count, int seed = 42)
    {
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));
        var rng = new Random(seed);
        var tickets = new[] { "POS-100", "AUTH-7", "API-314", "DB-42", "WEB-250", "MOB-8", "OPS-19", "INFRA-5", "UI-77", "PAY-12" };
        var dates = Enumerable.Range(0, 30).Select(i => new DateTime(2026, 7, 1).AddDays(i).ToString("yyyy-MM-dd")).ToArray();
        const string system = "You are a worklog assistant. When the user describes their work, call the LogTime tool with:\n- ticket: the ticket ID (e.g. POS-100, AUTH-7, API-314)\n- hours: decimal hours worked\n- date: ISO date string (use today's date if not specified)\n\nOffice hours rules:\n- The standard office day is 8 hours.\n- If the user says all day or full day, that means 8 hours.\n- Half day means 4 hours.\n- If the user mentions a specific duration, use that number.\n- Never interpret all day as 24 hours.\n\nIf the user mentions multiple tickets in one message, make a separate tool call for each ticket.\nIf the input is ambiguous, ask for clarification rather than guessing.";

        var examples = new List<WorklogSftExample>(count);
        for (int i = 0; i < count; i++)
        {
            int kind = rng.Next(6); // 0:explicit,1:all-day,2:half-day,3:multi,4:clarification,5:explicit-decimal
            string date = dates[rng.Next(dates.Length)];
            string ticket = tickets[rng.Next(tickets.Length)];
            string ticket2 = tickets[(Array.IndexOf(tickets, ticket) + 1 + rng.Next(tickets.Length - 1)) % tickets.Length];

            (string user, string assistant) = kind switch
            {
                0 => ($"I worked {Pick(rng, "2 hours", "3 hours", "6 hours")} on {ticket} on {date}.",
                      ToolCall(ticket, kind == 0 ? ParseHoursFromUser(rng, 2, 3, 6) : 2m, date)),
                1 => ($"I spent {Pick(rng, "all day", "the full day")} on {ticket} on {date}.",
                      ToolCall(ticket, 8m, date)),
                2 => ($"I did a {Pick(rng, "half day", "half-day")} on {ticket} on {date}.",
                      ToolCall(ticket, 4m, date)),
                3 => MultiTicketExample(rng, ticket, ticket2, date),
                4 => ClarificationExample(rng),
                _ => DecimalExample(rng, ticket, date),
            };

            // Fix kind 0 hours to match text
            if (kind == 0)
            {
                decimal h = user.Contains("2 hours") ? 2m : user.Contains("3 hours") ? 3m : 6m;
                assistant = ToolCall(ticket, h, date);
            }

            examples.Add(new WorklogSftExample([
                new WorklogMessage("system", system),
                new WorklogMessage("user", user),
                new WorklogMessage("assistant", assistant),
            ]));
        }

        return examples;
    }

    static (string user, string assistant) MultiTicketExample(Random rng, string t1, string t2, string date)
    {
        if (rng.Next(2) == 0)
        {
            return ($"4 hours on {t1} and 4 hours on {t2} on {date}.",
                ToolCall(t1, 4m, date) + "\n" + ToolCall(t2, 4m, date));
        }
        return ($"I split the day between {t1} and {t2} on {date}, half day each.",
            ToolCall(t1, 4m, date) + "\n" + ToolCall(t2, 4m, date));
    }

    static (string user, string assistant) DecimalExample(Random rng, string ticket, string date)
    {
        var options = new (string text, decimal hours)[] { ("2.5 hours", 2.5m), ("1.5 hours", 1.5m), ("0.5 hours", 0.5m), ("20 minutes", 0.33m) };
        var pick = options[rng.Next(options.Length)];
        return ($"I spent {pick.text} on {ticket} on {date}.", ToolCall(ticket, pick.hours, date));
    }

    static (string user, string assistant) ClarificationExample(Random rng)
    {
        string[] prompts = [
            "I worked on stuff today.",
            "Log my time please.",
            "I was busy all day.",
            "Worked on the ticket for a bit.",
        ];
        string[] replies = [
            "Could you clarify which ticket you worked on and how many hours to log?",
            "Which ticket should I log, and for how many hours? Please also provide the date if not today.",
            "Could you clarify which ticket you worked on? If you meant all day, I can log 8 hours once you specify the ticket and date.",
            "Could you specify the ticket ID and the number of hours you worked?",
        ];
        int i = rng.Next(prompts.Length);
        return (prompts[i], replies[i]);
    }

    static decimal ParseHoursFromUser(Random rng, params int[] choices) => choices[rng.Next(choices.Length)];

    static string Pick(Random rng, params string[] choices) => choices[rng.Next(choices.Length)];

    static string ToolCall(string ticket, decimal hours, string date)
    {
        string hoursText = hours % 1 == 0 ? ((int)hours).ToString(System.Globalization.CultureInfo.InvariantCulture) : hours.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return $"<tool_call>\n<function>LogTime</function>\n<parameter=ticket>{ticket}</parameter>\n<parameter=hours>{hoursText}</parameter>\n<parameter=date>{date}</parameter>\n</tool_call>";
    }
}
