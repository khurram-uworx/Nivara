using NivaraFineTuning;
using NUnit.Framework;

namespace Nivara.Tests.FineTuning;

[TestFixture]
public sealed class WorklogSftEvaluatorTests
{
    [Test]
    public void ParseLogTimeCalls_SingleCall_ParsesAllFields()
    {
        var text = "<tool_call>\n<function>LogTime</function>\n<parameter=ticket>POS-100</parameter>\n<parameter=hours>2.5</parameter>\n<parameter=date>2026-07-19</parameter>\n</tool_call>";
        var calls = WorklogSftEvaluator.ParseLogTimeCalls(text);

        Assert.That(calls.Count, Is.EqualTo(1));
        Assert.That(calls[0].Ticket, Is.EqualTo("POS-100"));
        Assert.That(calls[0].Hours, Is.EqualTo(2.5m));
        Assert.That(calls[0].Date, Is.EqualTo("2026-07-19"));
    }

    [Test]
    public void ParseLogTimeCalls_MultiCall_ParsesBoth()
    {
        var text = "<tool_call>\n<function>LogTime</function>\n<parameter=ticket>A</parameter>\n<parameter=hours>4</parameter>\n<parameter=date>2026-07-19</parameter>\n</tool_call>\n<tool_call>\n<function>LogTime</function>\n<parameter=ticket>B</parameter>\n<parameter=hours>4</parameter>\n<parameter=date>2026-07-19</parameter>\n</tool_call>";
        var calls = WorklogSftEvaluator.ParseLogTimeCalls(text);

        Assert.That(calls.Count, Is.EqualTo(2));
        Assert.That(calls.Select(c => c.Ticket), Is.EquivalentTo(new[] { "A", "B" }));
    }

    [Test]
    public void ParseLogTimeCalls_MalformedOrMissing_ReturnsEmpty()
    {
        Assert.That(WorklogSftEvaluator.ParseLogTimeCalls(""), Is.Empty);
        Assert.That(WorklogSftEvaluator.ParseLogTimeCalls("no tool call here"), Is.Empty);
        Assert.That(WorklogSftEvaluator.ParseLogTimeCalls("<tool_call><function>LogTime</function><parameter=ticket>X</parameter></tool_call>"), Is.Empty);
    }

    [Test]
    public void IsClarification_DetectsClarificationPhrases()
    {
        Assert.That(WorklogSftEvaluator.IsClarification("Could you clarify which ticket?"), Is.True);
        Assert.That(WorklogSftEvaluator.IsClarification("Which ticket should I log, and for how many hours?"), Is.True);
        Assert.That(WorklogSftEvaluator.IsClarification("<tool_call>...</tool_call>"), Is.False);
    }

    [Test]
    public void Evaluate_PerfectPrediction_ScoresAllCorrect()
    {
        var example = new WorklogSftExample(new List<WorklogMessage>
        {
            new("system", "sys"),
            new("user", "I worked 2 hours on POS-100 on 2026-07-19."),
            new("assistant", "<tool_call>\n<function>LogTime</function>\n<parameter=ticket>POS-100</parameter>\n<parameter=hours>2</parameter>\n<parameter=date>2026-07-19</parameter>\n</tool_call>")
        });

        var metrics = WorklogSftEvaluator.Evaluate(new[] { example }, _ => example.Messages.Last(m => m.Role == "assistant").Content);

        Assert.That(metrics.Total, Is.EqualTo(1));
        Assert.That(metrics.ValidStructure, Is.EqualTo(1));
        Assert.That(metrics.TicketCorrect, Is.EqualTo(1));
        Assert.That(metrics.HoursCorrect, Is.EqualTo(1));
        Assert.That(metrics.DateCorrect, Is.EqualTo(1));
        Assert.That(metrics.ValidStructureRate, Is.EqualTo(1.0));
    }

    [Test]
    public void Evaluate_MultiTicket_RequiresExactCount()
    {
        var example = new WorklogSftExample(new List<WorklogMessage>
        {
            new("system", "sys"),
            new("user", "4 hours on A and 4 hours on B on 2026-07-19."),
            new("assistant", "<tool_call>\n<function>LogTime</function>\n<parameter=ticket>A</parameter>\n<parameter=hours>4</parameter>\n<parameter=date>2026-07-19</parameter>\n</tool_call>\n<tool_call>\n<function>LogTime</function>\n<parameter=ticket>B</parameter>\n<parameter=hours>4</parameter>\n<parameter=date>2026-07-19</parameter>\n</tool_call>")
        });

        // Prediction drops one call -> not exact count -> not scored.
        var partial = "<tool_call>\n<function>LogTime</function>\n<parameter=ticket>A</parameter>\n<parameter=hours>4</parameter>\n<parameter=date>2026-07-19</parameter>\n</tool_call>";
        var metrics = WorklogSftEvaluator.Evaluate(new[] { example }, _ => partial);

        Assert.That(metrics.MultiTicketTotal, Is.EqualTo(1));
        Assert.That(metrics.MultiTicketCorrect, Is.EqualTo(0));
    }

    [Test]
    public void Evaluate_ConventionAllDay_ScoresConvention()
    {
        var example = new WorklogSftExample(new List<WorklogMessage>
        {
            new("system", "sys"),
            new("user", "I spent all day on POS-100 on 2026-07-19."),
            new("assistant", "<tool_call>\n<function>LogTime</function>\n<parameter=ticket>POS-100</parameter>\n<parameter=hours>8</parameter>\n<parameter=date>2026-07-19</parameter>\n</tool_call>")
        });

        var metrics = WorklogSftEvaluator.Evaluate(new[] { example }, _ => example.Messages.Last(m => m.Role == "assistant").Content);

        Assert.That(metrics.ConventionTotal, Is.EqualTo(1));
        Assert.That(metrics.ConventionCorrect, Is.EqualTo(1));
    }

    [Test]
    public void Evaluate_Clarification_ScoresClarification()
    {
        var example = new WorklogSftExample(new List<WorklogMessage>
        {
            new("system", "sys"),
            new("user", "I worked on stuff today."),
            new("assistant", "Could you clarify which ticket you worked on and how many hours to log?")
        });

        var metrics = WorklogSftEvaluator.Evaluate(new[] { example }, _ => "Could you clarify which ticket you worked on?");

        Assert.That(metrics.ClarificationTotal, Is.EqualTo(1));
        Assert.That(metrics.ClarificationCorrect, Is.EqualTo(1));
        Assert.That(metrics.ValidStructure, Is.EqualTo(1));
    }
}