using Nivara.Samples.Incident;
using NUnit.Framework;

namespace Nivara.Tests.Incident;

[TestFixture]
public class ScenarioTests
{
    [Test]
    public void Scenarios_All_ReturnsFourScenarios()
    {
        Assert.That(Scenarios.All, Has.Count.EqualTo(4));
    }

    [TestCase("A")]
    [TestCase("B")]
    [TestCase("C")]
    [TestCase("D")]
    public void Scenarios_Get_ReturnsCorrectScenario(string id)
    {
        var scenario = Scenarios.Get(id);
        Assert.That(scenario, Is.Not.Null);
        Assert.That(scenario.Id, Is.EqualTo(id));
    }

    [TestCase("a")]
    [TestCase("A")]
    public void Scenarios_Get_IsCaseInsensitive(string id)
    {
        var scenario = Scenarios.Get(id);
        Assert.That(scenario.Id, Is.EqualTo(id.ToUpperInvariant()));
    }

    [Test]
    public void Scenarios_Get_ThrowsOnUnknown()
    {
        Assert.Throws<ArgumentException>(() => Scenarios.Get("X"));
    }

    [Test]
    public void ScenarioA_DatabaseDegradation_HasCorrectServices()
    {
        var scenario = Scenarios.A;
        Assert.That(scenario.Name, Is.EqualTo("Database degradation"));
        Assert.That(scenario.AffectedServices, Does.Contain("orders"));
        Assert.That(scenario.AffectedServices, Does.Contain("gateway"));
    }

    [Test]
    public void ScenarioB_BadDeployment_HasDeployEvent()
    {
        var scenario = Scenarios.B;
        Assert.That(scenario.Events, Has.Count.GreaterThanOrEqualTo(1));
        Assert.That(scenario.Events.Any(e => e.EventType == "deploy"), Is.True);
    }

    [Test]
    public void ScenarioD_RegionalFailure_HasRegionalEvent()
    {
        var scenario = Scenarios.D;
        Assert.That(scenario.Events.Any(e => e.EventType == "regional_degradation"), Is.True);
    }

    [Test]
    public void AllScenarios_IncidentStartBeforeIncidentEnd()
    {
        foreach (var scenario in Scenarios.All)
        {
            Assert.That(scenario.IncidentStart, Is.LessThan(scenario.IncidentEnd),
                $"Scenario {scenario.Id}: IncidentStart should be before IncidentEnd");
        }
    }

    [Test]
    public void AllScenarios_HaveNonEmptyEvents()
    {
        foreach (var scenario in Scenarios.All)
        {
            Assert.That(scenario.Events.Count, Is.GreaterThan(0),
                $"Scenario {scenario.Id} should have at least one event");
        }
    }

    [Test]
    public void Scenarios_AreDeterministic()
    {
        var a1 = Scenarios.A;
        var a2 = Scenarios.Get("A");
        Assert.That(a1.IncidentStart, Is.EqualTo(a2.IncidentStart));
        Assert.That(a1.IncidentEnd, Is.EqualTo(a2.IncidentEnd));
        Assert.That(a1.Events.Count, Is.EqualTo(a2.Events.Count));
    }
}
