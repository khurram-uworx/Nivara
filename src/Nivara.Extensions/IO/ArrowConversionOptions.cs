using System.Text;

namespace Nivara.IO;

/// <summary>
/// Configuration options for Arrow conversion operations
/// </summary>
public class ArrowConversionOptions
{
    /// <summary>
    /// Gets or sets whether to use zero-copy operations when possible
    /// </summary>
    /// <remarks>
    /// When enabled, the converter will attempt to share memory between Nivara and Arrow structures
    /// to avoid copying data. Falls back to copying when zero-copy is not possible.
    /// Default is true.
    /// </remarks>
    public bool UseZeroCopy { get; set; } = true;

    /// <summary>
    /// Gets or sets the timezone for timestamp conversions
    /// </summary>
    /// <remarks>
    /// Used when converting DateTime values to/from Arrow timestamp types.
    /// Default is UTC.
    /// </remarks>
    public TimeZoneInfo TimeZone { get; set; } = TimeZoneInfo.Utc;

    /// <summary>
    /// Gets or sets whether to validate types during conversion
    /// </summary>
    /// <remarks>
    /// When enabled, performs additional type validation checks during conversion.
    /// Can be disabled for performance-critical scenarios.
    /// Default is true.
    /// </remarks>
    public bool ValidateTypes { get; set; } = true;

    /// <summary>
    /// Gets or sets the string encoding for text data
    /// </summary>
    /// <remarks>
    /// Specifies the encoding to use for string data during conversion.
    /// Default is UTF-8.
    /// </remarks>
    public Encoding StringEncoding { get; set; } = Encoding.UTF8;
}
