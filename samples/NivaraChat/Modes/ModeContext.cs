namespace NivaraChat.Modes;

/// <summary>
/// Carries the shared options (CLI flags or interactive-menu answers) into every mode
/// runner. Modeled after the option bags used by <c>TransformerMode</c>/<c>SmollmMode</c>.
/// </summary>
public sealed record ModeContext(
    string ModelsDir,
    string OllamaUrl,
    string ModelName,
    bool UseOllama,
    string? SingleShotText,
    float ConfidenceThreshold,
    string? DocsDir,
    int TopK,
    bool UseSilk = false,
    string? AclGroup = null,
    bool ShowRoundTrip = false,
    string Store = "frame",
    string? Index = null,
    bool Rebuild = false,
    bool NoCache = false,
    int BatchSize = 32,
    int ChunkSize = 500,
    bool Explore = false,
    string? ReportPath = null,
    string? Weights = null,
    bool Offline = false,
    int Runs = 3,
    double Temperature = 0.2,
    bool GradeAll = false,
    bool Verbose = false,
    string? Queries = null);
