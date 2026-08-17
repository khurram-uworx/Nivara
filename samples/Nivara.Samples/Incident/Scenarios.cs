namespace Nivara.Samples.Incident;

public sealed class ServiceEvent
{
    public DateTimeOffset Timestamp { get; init; }
    public string Service { get; init; } = "";
    public string EventType { get; init; } = "";
    public double Magnitude { get; init; }
}

public sealed class IncidentScenario
{
    public string Id { get; init; } = "";
    public string Name { get; init; } = "";
    public DateTimeOffset IncidentStart { get; init; }
    public DateTimeOffset IncidentEnd { get; init; }
    public IReadOnlyList<ServiceEvent> Events { get; init; } = [];
    public IReadOnlyList<string> AffectedServices { get; init; } = [];
}

public static class Scenarios
{
    static readonly DateTimeOffset BaseTime = new(2025, 6, 15, 14, 0, 0, TimeSpan.Zero);

    public static IncidentScenario A { get; } = new()
    {
        Id = "A",
        Name = "Database degradation",
        IncidentStart = BaseTime.AddMinutes(5),
        IncidentEnd = BaseTime.AddMinutes(25),
        Events =
        [
            new() { Timestamp = BaseTime.AddMinutes(5), Service = "orders", EventType = "latency_spike", Magnitude = 5.0 },
            new() { Timestamp = BaseTime.AddMinutes(7), Service = "checkout", EventType = "latency_spike", Magnitude = 3.0 },
            new() { Timestamp = BaseTime.AddMinutes(9), Service = "payments", EventType = "timeout_spike", Magnitude = 4.0 },
            new() { Timestamp = BaseTime.AddMinutes(10), Service = "gateway", EventType = "retry_storm", Magnitude = 8.0 },
        ],
        AffectedServices = ["orders", "checkout", "payments", "gateway"]
    };

    public static IncidentScenario B { get; } = new()
    {
        Id = "B",
        Name = "Bad deployment",
        IncidentStart = BaseTime.AddMinutes(17),
        IncidentEnd = BaseTime.AddMinutes(30),
        Events =
        [
            new() { Timestamp = BaseTime.AddMinutes(17), Service = "orders", EventType = "deploy", Magnitude = 1.0 },
            new() { Timestamp = BaseTime.AddMinutes(19), Service = "orders", EventType = "exception_spike", Magnitude = 6.0 },
            new() { Timestamp = BaseTime.AddMinutes(21), Service = "checkout", EventType = "failure_spike", Magnitude = 3.0 },
        ],
        AffectedServices = ["orders", "checkout"]
    };

    public static IncidentScenario C { get; } = new()
    {
        Id = "C",
        Name = "Traffic spike",
        IncidentStart = BaseTime.AddMinutes(10),
        IncidentEnd = BaseTime.AddMinutes(20),
        Events =
        [
            new() { Timestamp = BaseTime.AddMinutes(10), Service = "gateway", EventType = "traffic_multiplier", Magnitude = 8.0 },
            new() { Timestamp = BaseTime.AddMinutes(12), Service = "orders", EventType = "queue_depth_rise", Magnitude = 4.0 },
            new() { Timestamp = BaseTime.AddMinutes(14), Service = "payments", EventType = "latency_spike", Magnitude = 3.0 },
        ],
        AffectedServices = ["gateway", "orders", "payments", "checkout", "catalog"]
    };

    public static IncidentScenario D { get; } = new()
    {
        Id = "D",
        Name = "Regional failure",
        IncidentStart = BaseTime.AddMinutes(8),
        IncidentEnd = BaseTime.AddMinutes(22),
        Events =
        [
            new() { Timestamp = BaseTime.AddMinutes(8), Service = "gateway", EventType = "regional_degradation", Magnitude = 5.0 },
        ],
        AffectedServices = ["gateway", "orders", "payments", "checkout"]
    };

    public static IReadOnlyList<IncidentScenario> All { get; } = [A, B, C, D];

    public static IncidentScenario Get(string id) => id.ToUpperInvariant() switch
    {
        "A" => A,
        "B" => B,
        "C" => C,
        "D" => D,
        _ => throw new ArgumentException($"Unknown scenario: {id}")
    };
}
