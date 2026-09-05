using System.ComponentModel;

namespace NivaraChat.Qwen;

/// <summary>
/// Deterministic weather tool used by the <c>--qwen tools-weather</c> demo. The tool declaration
/// (name, description, parameter schema) is byte-identical to the Torch reference definition used
/// to produce <c>qwen_tool_prompt.txt</c> / <c>qwen_tool_final_prompt.txt</c>, so the C# renderer's
/// <c>tools</c> system block can be pinned against those fixtures.
/// </summary>
internal static class QwenSampleTools
{
    /// <summary>Maps the exact tool name the model is trained to emit.</summary>
    public const string WeatherToolName = "getWeather";

    /// <summary>
    /// Gets the current weather for a city. Returns a short description like 'Sunny, 22°C'.
    /// The Paris result mirrors the Torch fixture's tool response exactly.
    /// </summary>
    [Description("Gets the current weather for a city. Returns a short description like 'Sunny, 22°C'.")]
    public static string GetWeather(
        [Description("The city name, e.g. 'Paris' or 'New York'")] string city)
    {
        ArgumentNullException.ThrowIfNull(city);
        var normalized = city.Trim();
        if (normalized.Equals("Paris", StringComparison.OrdinalIgnoreCase))
            return "Partly cloudy, 18°C. Light breeze from the northwest.";
        return $"Partly cloudy, {12 + Math.Abs(normalized.Length) % 10}°C. Light breeze from the northwest.";
    }

    /// <summary>Builds the registered AIFunction for the demo tool loop.</summary>
    public static Microsoft.Extensions.AI.AIFunction CreateWeatherTool()
        => Microsoft.Extensions.AI.AIFunctionFactory.Create(
            (Func<string, string>)GetWeather,
            name: WeatherToolName);
}