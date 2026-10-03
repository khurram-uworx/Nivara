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
    public void Scenarios_Get_ThrowsArgumentNullOnNull()
    {
        // A null id is a caller error, not an unknown scenario. Without the guard this surfaced as
        // NullReferenceException from ToUpperInvariant, which reads like a bug inside Get.
        var ex = Assert.Throws<ArgumentNullException>(() => Scenarios.Get(null!));
        Assert.That(ex!.ParamName, Is.EqualTo("id"));
    }

    [Test]
    public void Scenarios_Get_ThrowsArgumentOnEmpty()
    {
        // An empty id is an unknown scenario, not a null one, so it must not be reported as
        // ArgumentNullException -- Assert.Throws matches the exact type, so this pins that down.
        var ex = Assert.Throws<ArgumentException>(() => Scenarios.Get(""));
        Assert.That(ex!.ParamName, Is.Null);
    }

    [Test]
    public void AllScenarios_EventsArePopulatedAndInsideTheIncidentWindow()
    {
        // Coverage: every event of every scenario. The guards make that non-vacuous -- without
        // them an empty Scenarios.All, or a scenario with no events, would pass this test having
        // asserted nothing at all.
        Assert.That(Scenarios.All, Is.Not.Empty, "no scenarios to check");
        var eventCount = 0;

        foreach (var scenario in Scenarios.All)
        {
            Assert.That(scenario.Events, Is.Not.Empty, $"scenario {scenario.Id} has no events to check");
            eventCount += scenario.Events.Count;

            foreach (var evt in scenario.Events)
            {
                var where = $"Scenario {scenario.Id} event '{evt.EventType}'";
                Assert.Multiple(() =>
                {
                    Assert.That(evt.Service, Is.Not.Empty, $"{where}: Service should name a service");
                    Assert.That(evt.EventType, Is.Not.Empty, $"{where}: EventType should name a type");
                    Assert.That(evt.Magnitude, Is.GreaterThan(0), $"{where}: Magnitude should be positive");
                    Assert.That(evt.Timestamp, Is.GreaterThanOrEqualTo(scenario.IncidentStart),
                        $"{where}: Timestamp {evt.Timestamp:O} precedes IncidentStart {scenario.IncidentStart:O}");
                    Assert.That(evt.Timestamp, Is.LessThanOrEqualTo(scenario.IncidentEnd),
                        $"{where}: Timestamp {evt.Timestamp:O} follows IncidentEnd {scenario.IncidentEnd:O}");
                    Assert.That(scenario.AffectedServices, Does.Contain(evt.Service),
                        $"{where}: Service '{evt.Service}' is not in AffectedServices");
                });
            }
        }

        Assert.That(eventCount, Is.EqualTo(Scenarios.All.Sum(s => s.Events.Count)),
            "every event should have been visited exactly once");
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
