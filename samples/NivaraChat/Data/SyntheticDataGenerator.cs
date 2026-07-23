using System.Text;

namespace NivaraChat.Data;

public static class SyntheticDataGenerator
{
    static readonly string[] PersonNames = ["John Smith", "Jane Doe", "Bob Wilson", "Alice Brown", "Charlie Davis", "Emma Johnson", "Michael Lee", "Sarah Garcia"];
    static readonly string[] OrgNames = ["Acme Corp", "TechStart Inc", "Global Industries", "MegaSoft", "InnovateLab", "DataFlow Systems", "CloudNine Ltd", "NetWave"];
    static readonly string[] Dates = ["January 15", "March 3", "June 20", "September 8", "December 25", "next Monday", "end of month", "Q3 2026"];
    static readonly string[] Locations = ["New York", "London", "Tokyo", "San Francisco", "Berlin", "Sydney", "Toronto", "Singapore"];
    static readonly string[] PositivePhrases = ["great work", "excellent results", "on track", "ahead of schedule", "exceeding expectations", "strong performance"];
    static readonly string[] NegativePhrases = ["needs improvement", "behind schedule", "below targets", "concerns raised", "issues found", "declining metrics"];
    static readonly string[] NeutralPhrases = ["status update", "regular review", "ongoing monitoring", "standard procedure", "routine check", "scheduled assessment"];
    static readonly string[] Activities = ["completed the project", "submitted the report", "reviewed the proposal", "updated the system", "resolved the issue", "analyzed the data"];

    public static (string[] texts, int[] labels) GenerateSentimentData(int count, int seed = 42)
    {
        var rng = new Random(seed);
        var texts = new string[count];
        var labels = new int[count];

        for (int i = 0; i < count; i++)
        {
            int sentiment = rng.Next(3);
            string text = sentiment switch
            {
                0 => GenerateNegativeSentence(rng),
                1 => GenerateNeutralSentence(rng),
                _ => GeneratePositiveSentence(rng)
            };
            texts[i] = text;
            labels[i] = sentiment;
        }
        return (texts, labels);
    }

    public static (string[] texts, string[] labelSequences) GenerateEntityData(int count, int seed = 42)
    {
        var rng = new Random(seed);
        var texts = new string[count];
        var labelSequences = new string[count];

        for (int i = 0; i < count; i++)
        {
            var (text, labels) = GenerateEntitySentence(rng);
            texts[i] = text;
            labelSequences[i] = labels;
        }
        return (texts, labelSequences);
    }

    public static (string[] inputs, int[] labels) GenerateValidatorData(int count, int seed = 42)
    {
        var rng = new Random(seed);
        var inputs = new string[count];
        var labels = new int[count];

        for (int i = 0; i < count; i++)
        {
            bool consistent = rng.Next(2) == 0;
            var (original, entities) = GenerateEntitySentence(rng);
            string response = consistent
                ? GenerateConsistentResponse(entities, rng)
                : GenerateInconsistentResponse(rng);
            inputs[i] = original + " || " + response;
            labels[i] = consistent ? 1 : 0;
        }
        return (inputs, labels);
    }

    public static (string[] inputs, int[] labels) GenerateAgentsValidatorData(int count, int seed = 42)
    {
        var rng = new Random(seed);
        var inputs = new string[count];
        var labels = new int[count];

        for (int i = 0; i < count; i++)
        {
            bool consistent = rng.Next(2) == 0;
            inputs[i] = GenerateAgentsPipelineOutput(rng, consistent);
            labels[i] = consistent ? 1 : 0;
        }
        return (inputs, labels);
    }

    static string GenerateAgentsPipelineOutput(Random rng, bool consistent)
    {
        var sentimentLabels = new[] { "Positive", "Negative", "Neutral" };
        var sentiment = Pick(rng, sentimentLabels);
        float confidence = consistent ? (float)(0.7 + rng.NextDouble() * 0.3) : (float)(0.1 + rng.NextDouble() * 0.3);

        string entityLine;
        if (consistent)
        {
            var (original, entities) = GenerateEntitySentence(rng);
            var detectedEntities = entities.Split(' ')
                .Where(l => l != "O")
                .Select(l => l.Replace("B-", "").ToLower())
                .Distinct()
                .ToList();

            var entityDict = new Dictionary<string, List<string>>
            {
                ["person"] = [], ["org"] = [], ["date"] = [], ["location"] = []
            };

            var tokens = original.Split(' ');
            var entityClasses = new[] { "O", "B-person", "B-org", "B-date", "B-location" };
            var wordToEntity = BuildEntityLookup();

            foreach (var token in tokens)
            {
                if (wordToEntity.TryGetValue(token, out var label) && label != "O")
                {
                    var entityType = label.Replace("B-", "").ToLower();
                    if (entityDict.ContainsKey(entityType))
                        entityDict[entityType].Add(token.ToLower());
                }
            }

            entityLine = System.Text.Json.JsonSerializer.Serialize(entityDict);
        }
        else
        {
            var failureTypes = new[]
            {
                "Unable to extract entities (confidence: 0.20)\n{\"person\":[],\"org\":[],\"date\":[],\"location\":[]}",
                "{\"person\":[],\"org\":[],\"date\":[],\"location\":[]}",
                "Error: model output truncated",
                ""
            };
            entityLine = Pick(rng, failureTypes);
        }

        return $"{sentiment} (confidence: {confidence:F2})\n{entityLine}";
    }

    static Dictionary<string, string> BuildEntityLookup()
    {
        var lookup = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var name in PersonNames)
            foreach (var word in name.Split(' '))
                lookup[word] = "B-person";
        foreach (var org in OrgNames)
            foreach (var word in org.Split(' '))
                lookup[word] = "B-org";
        foreach (var date in Dates)
            foreach (var word in date.Split(' '))
                lookup[word] = "B-date";
        foreach (var loc in Locations)
            foreach (var word in loc.Split(' '))
                lookup[word] = "B-location";
        return lookup;
    }

    static string GeneratePositiveSentence(Random rng)
    {
        var person = Pick(rng, PersonNames);
        var org = Pick(rng, OrgNames);
        var activity = Pick(rng, PositivePhrases);
        return $"{person} from {org} reported {activity} this quarter";
    }

    static string GenerateNegativeSentence(Random rng)
    {
        var person = Pick(rng, PersonNames);
        var org = Pick(rng, OrgNames);
        var activity = Pick(rng, NegativePhrases);
        return $"{person} at {org} noted {activity} in the recent review";
    }

    static string GenerateNeutralSentence(Random rng)
    {
        var person = Pick(rng, PersonNames);
        var org = Pick(rng, OrgNames);
        var activity = Pick(rng, NeutralPhrases);
        return $"{person} from {org} provided a {activity} during the meeting";
    }

    static (string text, string labels) GenerateEntitySentence(Random rng)
    {
        var templates = new[]
        {
            () => {
                var person = Pick(rng, PersonNames);
                var org = Pick(rng, OrgNames);
                var date = Pick(rng, Dates);
                var text = $"{person} from {org} on {date}";
                return LabelEntities(text);
            },
            () => {
                var org = Pick(rng, OrgNames);
                var loc = Pick(rng, Locations);
                var date = Pick(rng, Dates);
                var text = $"{org} in {loc} announced on {date}";
                return LabelEntities(text);
            },
            () => {
                var person = Pick(rng, PersonNames);
                var loc = Pick(rng, Locations);
                var text = $"{person} will visit {loc} next week";
                return LabelEntities(text);
            }
        };

        return Pick(rng, templates)();
    }

    static (string text, string labels) LabelEntities(string text)
    {
        var wordToEntity = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var name in PersonNames)
            foreach (var word in name.Split(' '))
                wordToEntity[word] = "B-person";
        foreach (var org in OrgNames)
            foreach (var word in org.Split(' '))
                wordToEntity[word] = "B-org";
        foreach (var date in Dates)
            foreach (var word in date.Split(' '))
                wordToEntity[word] = "B-date";
        foreach (var loc in Locations)
            foreach (var word in loc.Split(' '))
                wordToEntity[word] = "B-location";

        var tokens = text.Split(' ');
        var lbs = new List<string>();
        foreach (var t in tokens)
        {
            if (wordToEntity.TryGetValue(t, out var label))
                lbs.Add(label);
            else
                lbs.Add("O");
        }
        return (string.Join(' ', tokens), string.Join(' ', lbs));
    }

    static string GenerateConsistentResponse(string entityLabels, Random rng)
    {
        var entities = entityLabels.Split(' ')
            .Where(l => l != "O")
            .Select(l => l.Replace("B-", ""))
            .Distinct()
            .ToArray();

        if (entities.Length == 0)
            return "The information has been reviewed and confirmed.";

        var sb = new StringBuilder("Confirmed: ");
        sb.Append(string.Join(", ", entities.Take(2)));
        sb.Append(" noted in the record.");
        return sb.ToString();
    }

    static string GenerateInconsistentResponse(Random rng)
    {
        var wrongEntities = new[] { "unknown entity", "unverified source", "pending confirmation", "no record found" };
        return $"Review complete. {Pick(rng, wrongEntities)}. No further action needed.";
    }

    static T Pick<T>(Random rng, T[] array) => array[rng.Next(array.Length)];

    public static void SaveSentimentCsv(string path, string[] texts, int[] labels)
    {
        var sb = new StringBuilder();
        sb.AppendLine("text,label");
        for (int i = 0; i < texts.Length; i++)
            sb.AppendLine($"\"{texts[i].Replace("\"", "\"\"")}\",{labels[i]}");
        File.WriteAllText(path, sb.ToString());
    }

    public static void SaveEntityCsv(string path, string[] texts, string[] labels)
    {
        var sb = new StringBuilder();
        sb.AppendLine("text,labels");
        for (int i = 0; i < texts.Length; i++)
            sb.AppendLine($"\"{texts[i].Replace("\"", "\"\"")}\",\"{labels[i]}\"");
        File.WriteAllText(path, sb.ToString());
    }

    public static void SaveValidatorCsv(string path, string[] inputs, int[] labels)
    {
        var sb = new StringBuilder();
        sb.AppendLine("input,label");
        for (int i = 0; i < inputs.Length; i++)
            sb.AppendLine($"\"{inputs[i].Replace("\"", "\"\"")}\",{labels[i]}");
        File.WriteAllText(path, sb.ToString());
    }
}
