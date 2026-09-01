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
    int TopK);