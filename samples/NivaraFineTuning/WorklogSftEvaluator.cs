using System.Globalization;
using System.Text.RegularExpressions;

namespace NivaraFineTuning;

public sealed record WorklogLogTimeCall(string Ticket, decimal Hours, string Date);

public sealed record WorklogEvalCase(
    string UserText,
    string ExpectedAssistant,
    IReadOnlyList<WorklogLogTimeCall> ExpectedCalls,
    bool ExpectedClarification);

public sealed record WorklogEvalMetrics(
    int Total,
    int ValidStructure,
    int TicketCorrect,
    int HoursCorrect,
    int DateCorrect,
    int MultiTicketCorrect,
    int MultiTicketTotal,
    int ClarificationCorrect,
    int ClarificationTotal,
    int ConventionCorrect,
    int ConventionTotal)
{
    public double ValidStructureRate => Total == 0 ? 0 : (double)ValidStructure / Total;
    public double TicketAccuracy => Total == 0 ? 0 : (double)TicketCorrect / Total;
    public double HoursAccuracy => Total == 0 ? 0 : (double)HoursCorrect / Total;
    public double DateAccuracy => Total == 0 ? 0 : (double)DateCorrect / Total;
    public double MultiTicketAccuracy => MultiTicketTotal == 0 ? 1 : (double)MultiTicketCorrect / MultiTicketTotal;
    public double ClarificationAccuracy => ClarificationTotal == 0 ? 1 : (double)ClarificationCorrect / ClarificationTotal;
    public double ConventionAccuracy => ConventionTotal == 0 ? 1 : (double)ConventionCorrect / ConventionTotal;

    public string ToReport(string label)
    {
        return $"""
{label}
  total={Total}
  valid_structure={ValidStructure}/{Total} ({ValidStructureRate:P1})
  ticket={TicketCorrect}/{Total} ({TicketAccuracy:P1})
  hours={HoursCorrect}/{Total} ({HoursAccuracy:P1})
  date={DateCorrect}/{Total} ({DateAccuracy:P1})
  multi_ticket={MultiTicketCorrect}/{MultiTicketTotal} ({MultiTicketAccuracy:P1})
  clarification={ClarificationCorrect}/{ClarificationTotal} ({ClarificationAccuracy:P1})
  convention={ConventionCorrect}/{ConventionTotal} ({ConventionAccuracy:P1})
""";
    }
}

public static class WorklogSftEvaluator
{
    static readonly Regex ToolCallRegex = new(@"<tool_call>\s*<function>\s*LogTime\s*</function>(?<body>.*?)</tool_call>", RegexOptions.Singleline | RegexOptions.IgnoreCase | RegexOptions.Compiled);
    static readonly Regex ParamRegex = new(@"<parameter=(?<name>[^>]+)>(?<value>.*?)</parameter>", RegexOptions.Singleline | RegexOptions.IgnoreCase | RegexOptions.Compiled);

    public static IReadOnlyList<WorklogLogTimeCall> ParseLogTimeCalls(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return [];
        var calls = new List<WorklogLogTimeCall>();
        foreach (Match m in ToolCallRegex.Matches(text))
        {
            var body = m.Groups["body"].Value;
            string? ticket = null;
            string? hoursRaw = null;
            string? date = null;
            foreach (Match p in ParamRegex.Matches(body))
            {
                var name = p.Groups["name"].Value.Trim().ToLowerInvariant();
                var value = p.Groups["value"].Value.Trim();
                switch (name)
                {
                    case "ticket": ticket = value; break;
                    case "hours": hoursRaw = value; break;
                    case "date": date = value; break;
                }
            }
            if (ticket is null || hoursRaw is null || date is null) continue;
            if (!decimal.TryParse(hoursRaw, NumberStyles.Number, CultureInfo.InvariantCulture, out var hours)) continue;
            calls.Add(new WorklogLogTimeCall(ticket, hours, date));
        }
        return calls;
    }

    public static bool IsClarification(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return false;
        if (ParseLogTimeCalls(text).Count > 0) return false;
        var lower = text.ToLowerInvariant();
        return lower.Contains("clarif") || lower.Contains("which ticket") || lower.Contains("how many hours") || lower.Contains("what date") || lower.Contains("could you") || lower.Contains("can you specify");
    }

    public static WorklogEvalCase FromExample(WorklogSftExample ex)
    {
        var user = ex.Messages.LastOrDefault(m => string.Equals(m.Role, "user", StringComparison.OrdinalIgnoreCase))?.Content ?? "";
        var assistant = ex.Messages.LastOrDefault(m => string.Equals(m.Role, "assistant", StringComparison.OrdinalIgnoreCase))?.Content ?? "";
        var calls = ParseLogTimeCalls(assistant);
        bool clarification = calls.Count == 0 && IsClarification(assistant);
        return new WorklogEvalCase(user, assistant, calls, clarification);
    }

    public static WorklogEvalMetrics Evaluate(IReadOnlyList<WorklogSftExample> examples, Func<WorklogSftExample, string> predict)
    {
        ArgumentNullException.ThrowIfNull(examples);
        ArgumentNullException.ThrowIfNull(predict);

        int total = examples.Count;
        int valid = 0, ticket = 0, hours = 0, date = 0, multiOk = 0, multiTotal = 0, clarOk = 0, clarTotal = 0, convOk = 0, convTotal = 0;

        foreach (var ex in examples)
        {
            var expected = FromExample(ex);
            string predText = predict(ex) ?? "";
            var predCalls = ParseLogTimeCalls(predText);
            bool predClar = predCalls.Count == 0 && IsClarification(predText);

            bool structureOk = expected.ExpectedClarification ? predClar : predCalls.Count > 0;
            if (structureOk) valid++;

            if (expected.ExpectedClarification)
            {
                clarTotal++;
                if (predClar) clarOk++;
                continue;
            }

            // Compare calls as sets by ticket+hours+date for multi-ticket; simple for single.
            bool allTickets = expected.ExpectedCalls.All(ec => predCalls.Any(pc => string.Equals(pc.Ticket, ec.Ticket, StringComparison.OrdinalIgnoreCase)));
            bool allHours = expected.ExpectedCalls.All(ec => predCalls.Any(pc => pc.Hours == ec.Hours));
            bool allDates = expected.ExpectedCalls.All(ec => predCalls.Any(pc => string.Equals(pc.Date, ec.Date, StringComparison.OrdinalIgnoreCase)));
            bool exactCount = predCalls.Count == expected.ExpectedCalls.Count;

            if (allTickets && exactCount) ticket++;
            if (allHours && exactCount) hours++;
            if (allDates && exactCount) date++;

            if (expected.ExpectedCalls.Count > 1)
            {
                multiTotal++;
                if (exactCount && allTickets && allHours && allDates) multiOk++;
            }

            var convention = DetectConvention(expected.UserText);
            if (convention is not null)
            {
                convTotal++;
                if (ConventionSatisfied(convention, predCalls)) convOk++;
            }
        }

        return new WorklogEvalMetrics(total, valid, ticket, hours, date, multiOk, multiTotal, clarOk, clarTotal, convOk, convTotal);
    }

    static string? DetectConvention(string userText)
    {
        var lower = userText.ToLowerInvariant();
        if (lower.Contains("all day") || lower.Contains("full day")) return "all-day";
        if (lower.Contains("half day") || lower.Contains("half-day")) return "half-day";
        return null;
    }

    static bool ConventionSatisfied(string convention, IReadOnlyList<WorklogLogTimeCall> calls)
    {
        if (calls.Count == 0) return false;
        return convention switch
        {
            "all-day" => calls.All(c => c.Hours == 8m),
            "half-day" => calls.All(c => c.Hours == 4m),
            _ => true
        };
    }
}
