namespace Nivara.Samples.Incident;

public sealed record RequestTelemetry(
    DateTimeOffset Timestamp,
    string Service,
    string Endpoint,
    double DurationMs,
    int StatusCode,
    string Region,
    string TraceId,
    bool IsRetry);

public sealed record DeploymentEvent(
    DateTimeOffset Timestamp,
    string Service,
    string Version,
    string Region);

public sealed record ServiceDependency(
    string Parent,
    string Child);

public sealed record InstanceState(
    DateTimeOffset Timestamp,
    string Service,
    string InstanceId,
    string Region,
    int ActiveRequests,
    int QueueDepth);
