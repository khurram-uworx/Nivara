using NivaraFineTuning;
using NUnit.Framework;

namespace Nivara.Tests.FineTuning;

[TestFixture]
public sealed class WorklogSftEvaluatorTests
{
    [Test]
    public void ParseLogTimeCalls_SingleCall_ParsesFields()
    {
        const string text = "<tool_call>\n<function>LogTime</function>\n<parameter=ticket>POS-100</parameter>\n<parameter=hours>2.5</parameter>\n<parameter=date>2026-07-19</parameter>\n</tool_call>";

        var calls = WorklogSftEvaluator.ParseLogTimeCalls(text);

        Assert.That(calls, Has.Count.EqualTo(1));
        Assert.That(calls[0].Ticket, Is.EqualTo("POS-100"));
        Assert.That(calls[0].Hours, Is.EqualTo(2.5m));
        Assert.That(calls[0].Date, Is.EqualTo("2026-07-19"));
    }

    [Test]
    public void ParseLogTimeCalls_MultipleCalls_ParsesAll()
    {
        const string text = "<tool_call>\n<function>LogTime</function>\n<parameter=ticket>DB-42</parameter>\n<parameter=hours>4</parameter>\n<parameter=date>2026-07-19</parameter>\n</tool_call>\n" +
                            "<tool_call>\n<function>LogTime</function>\n<parameter=ticket>AUTH-7</parameter>\n<parameter=hours>4</parameter>\n<parameter=date>2026-07-19</parameter>\n</tool_call>";

        var calls = WorklogSftEvaluator.ParseLogTimeCalls(text);

        Assert.That(calls.Select(c => c.Ticket), Is.EqualTo(new[] { "DB-42", "AUTH-7" }));
        Assert.That(calls.All(c => c.Hours == 4m), Is.True);
    }

    [Test]
    public void ParseLogTimeCalls_MalformedOrMissingParams_IgnoresInvalid()
    {
        const string missingDate = "<tool_call><function>LogTime</function><parameter=ticket>POS-1</parameter><parameter=hours>1</parameter></tool_call>";
        const string badHours = "<tool_call><function>LogTime</function><parameter=ticket>POS-1</parameter><parameter=hours>abc</parameter><parameter=date>2026-07-19</parameter></tool_call>";

        Assert.That(WorklogSftEvaluator.ParseLogTimeCalls(missingDate), Is.Empty);
        Assert.That(WorklogSftEvaluator.ParseLogTimeCalls(badHours), Is.Empty);
    }

    [Test]
    public void IsClarification_ClarificationWithoutToolCall_ReturnsTrue()
    {
        Assert.That(WorklogSftEvaluator.IsClarification("Could you clarify which ticket you worked on?"), Is.True);
        Assert.That(WorklogSftEvaluator.IsClarification("Which ticket should I log, and for how many hours?"), Is.True);
    }

    [Test]
    public void IsClarification_ToolCallOrEmpty_ReturnsFalse()
    {
        const string toolCall = "<tool_call><function>LogTime</function><parameter=ticket>POS-1</parameter><parameter=hours>1</parameter><parameter=date>2026-07-19</parameter></tool_call>";
        Assert.That(WorklogSftEvaluator.IsClarification(toolCall), Is.False);
        Assert.That(WorklogSftEvaluator.IsClarification(""), Is.False);
    }

    [Test]
    public void Evaluate_PerfectPredictions_ReturnsPerfectMetrics()
    {
        var examples = new List<WorklogSftExample>
        {
            Example("I worked 2 hours on POS-100 on 2026-07-19.", ToolCall("POS-100", 2m, "2026-07-19")),
            Example("I spent all day on AUTH-7 on 2026-07-19.", ToolCall("AUTH-7", 8m, "2026-07-19")),
            Example("4 hours on DB-42 and 4 hours on AUTH-33 on 2026-07-19.", ToolCall("DB-42", 4m, "2026-07-19") + "\n" + ToolCall("AUTH-33", 4m, "2026-07-19")),
            Example("I worked on stuff today.", "Could you clarify which ticket you worked on and how many hours to log?")
        };

        var metrics = WorklogSftEvaluator.Evaluate(examples, ex => Assistant(ex));

        Assert.That(metrics.Total, Is.EqualTo(4));
        Assert.That(metrics.ValidStructure, Is.EqualTo(4));
        Assert.That(metrics.TicketCorrect, Is.EqualTo(3));
        Assert.That(metrics.HoursCorrect, Is.EqualTo(3));
        Assert.That(metrics.DateCorrect, Is.EqualTo(3));
        Assert.That(metrics.MultiTicketCorrect, Is.EqualTo(1));
        Assert.That(metrics.MultiTicketTotal, Is.EqualTo(1));
        Assert.That(metrics.ClarificationCorrect, Is.EqualTo(1));
        Assert.That(metrics.ClarificationTotal, Is.EqualTo(1));
        Assert.That(metrics.ConventionCorrect, Is.EqualTo(1));
        Assert.That(metrics.ConventionTotal, Is.EqualTo(1));
    }

    [Test]
    public void Evaluate_MultiTicketRequiresExactCountAndFields()
    {
        var expected = Example("4 hours on DB-42 and 4 hours on AUTH-33 on 2026-07-19.",
            ToolCall("DB-42", 4m, "2026-07-19") + "\n" + ToolCall("AUTH-33", 4m, "2026-07-19"));
        var examples = new List<WorklogSftExample> { expected };

        var missingOne = WorklogSftEvaluator.Evaluate(examples, _ => ToolCall("DB-42", 4m, "2026-07-19"));
        var wrongHours = WorklogSftEvaluator.Evaluate(examples, _ => ToolCall("DB-42", 4m, "2026-07-19") + "\n" + ToolCall("AUTH-33", 3m, "2026-07-19"));

        Assert.That(missingOne.MultiTicketCorrect, Is.EqualTo(0));
        Assert.That(missingOne.TicketCorrect, Is.EqualTo(0));
        Assert.That(wrongHours.MultiTicketCorrect, Is.EqualTo(0));
        Assert.That(wrongHours.HoursCorrect, Is.EqualTo(0));
    }

    [Test]
    public void Evaluate_ConventionScoring_RequiresExpectedHours()
    {
        var examples = new List<WorklogSftExample>
        {
            Example("I spent all day on POS-100 on 2026-07-19.", ToolCall("POS-100", 8m, "2026-07-19")),
            Example("I did a half day on AUTH-7 on 2026-07-19.", ToolCall("AUTH-7", 4m, "2026-07-19"))
        };

        var metrics = WorklogSftEvaluator.Evaluate(examples, ex => Assistant(ex).Replace("8", "24").Replace("4", "2", StringComparison.Ordinal));

        Assert.That(metrics.ConventionTotal, Is.EqualTo(2));
        Assert.That(metrics.ConventionCorrect, Is.EqualTo(0));
    }

    static WorklogSftExample Example(string user, string assistant) => new([
        new WorklogMessage("system", "sys"),
        new WorklogMessage("user", user),
        new WorklogMessage("assistant", assistant)
    ]);

    static string Assistant(WorklogSftExample ex) => ex.Messages.Last(m => m.Role == "assistant").Content;

    static string ToolCall(string ticket, decimal hours, string date)
    {
        var hoursText = hours % 1 == 0 ? ((int)hours).ToString(System.Globalization.CultureInfo.InvariantCulture) : hours.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return $"<tool_call>\n<function>LogTime</function>\n<parameter=ticket>{ticket}</parameter>\n<parameter=hours>{hoursText}</parameter>\n<parameter=date>{date}</parameter>\n</tool_call>";
    }
}
