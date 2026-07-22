namespace NivaraClassifier;

public static class DataGenerator
{
    static readonly string[] PositiveTemplates =
    [
        "i loved the {noun}",
        "the {noun} was absolutely fantastic",
        "great {noun} and wonderful {noun}",
        "amazing experience with the {noun}",
        "{noun} exceeded my expectations",
        "highly recommend this {noun}",
        "the {noun} is perfect for {activity}"
    ];

    static readonly string[] NegativeTemplates =
    [
        "i hated the {noun}",
        "the {noun} was terrible and boring",
        "awful {noun} complete waste of time",
        "very disappointed with the {noun}",
        "{noun} broke immediately",
        "do not buy this {noun}",
        "the {noun} is useless for {activity}"
    ];

    static readonly string[] Nouns =
    [
        "product", "movie", "book", "service", "restaurant", "hotel",
        "experience", "food", "phone", "laptop", "app", "game", "course",
        "concert", "album", "workout", "software", "website", "delivery",
        "cleaning", "coating", "charging", "wait", "room", "view",
        "location", "staff", "system", "tool", "package", "version",
        "update", "support", "quality", "design", "build", "battery",
        "screen", "speed", "taste", "price", "value"
    ];

    static readonly string[] Activities =
    [
        "daily use", "travel", "work", "gaming", "cooking",
        "fitness", "studying", "entertainment", "relaxation", "productivity",
        "streaming", "editing", "photography", "reading", "music",
        "sleeping", "outdoor", "commuting", "training", "teaching", "learning"
    ];

    static readonly string[] Adverbs =
    [
        "", "", "", "", "",
        "very", "really", "extremely", "somewhat", "quite"
    ];

    public static (string[] texts, int[] labels) Generate(int count, int seed)
    {
        var rng = new Random(seed);
        var texts = new string[count];
        var labels = new int[count];

        for (int i = 0; i < count; i++)
        {
            bool positive = rng.Next(2) == 1;
            var templates = positive ? PositiveTemplates : NegativeTemplates;
            string template = templates[rng.Next(templates.Length)];

            string text = template
                .Replace("{noun}", Nouns[rng.Next(Nouns.Length)])
                .Replace("{activity}", Activities[rng.Next(Activities.Length)]);

            string adverb = Adverbs[rng.Next(Adverbs.Length)];
            if (adverb.Length > 0)
            {
                var words = text.Split(' ');
                int insertPos = rng.Next(1, words.Length);
                var result = new List<string>(words);
                result.Insert(insertPos, adverb);
                text = string.Join(' ', result);
            }

            texts[i] = text;
            labels[i] = positive ? 1 : 0;
        }

        return (texts, labels);
    }

    public static void SaveCsv(string path, int count, int seed)
    {
        var (texts, labels) = Generate(count, seed);
        using var writer = new StreamWriter(path);
        writer.WriteLine("text,label");
        for (int i = 0; i < count; i++)
        {
            string escaped = texts[i].Replace("\"", "\"\"");
            writer.WriteLine($"\"{escaped}\",{labels[i]}");
        }
    }

    public static (string[] texts, int[] labels) LoadCsv(string path)
    {
        var lines = File.ReadAllLines(path)
            .Skip(1)
            .Where(l => !string.IsNullOrWhiteSpace(l))
            .ToArray();

        var texts = new string[lines.Length];
        var labels = new int[lines.Length];

        for (int i = 0; i < lines.Length; i++)
        {
            var parts = ParseCsvLine(lines[i]);
            texts[i] = parts[0];
            labels[i] = int.Parse(parts[1]);
        }

        return (texts, labels);
    }

    static string[] ParseCsvLine(string line)
    {
        var result = new List<string>();
        bool inQuotes = false;
        var current = new System.Text.StringBuilder();

        foreach (char c in line)
        {
            if (c == '"')
            {
                inQuotes = !inQuotes;
            }
            else if (c == ',' && !inQuotes)
            {
                result.Add(current.ToString());
                current.Clear();
            }
            else
            {
                current.Append(c);
            }
        }
        result.Add(current.ToString());

        return result.ToArray();
    }
}
