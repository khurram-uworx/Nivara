using Microsoft.Extensions.AI;
using System.ComponentModel;

namespace NivaraChat.SmolLM;

/// <summary>
/// Defines the deterministic <c>GetWeather</c> tool used to prove the Stage B native tool-calling
/// loop (SmolLM emits <c>&lt;tool_call&gt;</c> → <see cref="FunctionInvokingChatClient"/> invokes it
/// → the result is fed back as <c>&lt;tool_response&gt;</c>). Network-free and fully deterministic
/// so the pipeline is as predictable as possible before Nivara model tools arrive in Phase C.
/// </summary>
public static class SmollmTools
{
    [Description("Gets the current weather for a city. Returns a short description like 'Sunny, 22°C'.")]
    public static string GetWeather(
        [Description("The city name, e.g. 'Paris' or 'New York'")] string city)
    {
        ArgumentNullException.ThrowIfNull(city);

        return city.Trim().ToLowerInvariant() switch
        {
            "paris" => "Partly cloudy, 18°C. Light breeze from the northwest.",
            "london" => "Overcast with light rain, 14°C.",
            "new york" => "Sunny, 25°C. Clear skies expected.",
            "tokyo" => "Humid and warm, 28°C. Chance of afternoon showers.",
            "berlin" => "Cool and breezy, 12°C. Mostly cloudy.",
            _ => $"Clear skies, 20°C in {city}. Pleasant weather.",
        };
    }

    /// <summary>Creates the Stage B tool set (weather only) as <see cref="AIFunction"/>s.</summary>
    public static AITool[] GetWeatherTools()
        => [AIFunctionFactory.Create(GetWeather)];
}
