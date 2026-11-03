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
    /// <para>
    /// When enabled, the converter shares the column's existing data buffer with the resulting
    /// Apache.Arrow array instead of copying it. The validity bitmap is always rebuilt from the
    /// column's null mask, because a byte-wise, true-at-null mask cannot be shared with Arrow's
    /// bit-packed, true-at-valid bitmap.
    /// </para>
    /// <para>
    /// Eligible columns are flat columns of <c>byte</c>, <c>sbyte</c>, <c>short</c>,
    /// <c>ushort</c>, <c>int</c>, <c>uint</c>, <c>long</c>, <c>ulong</c>, <c>float</c>,
    /// <c>double</c> and <c>Half</c>, with or without nulls. Every other column - including
    /// <c>bool</c>, whose values Arrow packs as bits, and <c>string</c> - throws
    /// <see cref="UnsupportedTypeException"/> naming the column rather than silently copying.
    /// </para>
    /// <para>
    /// Export only: <c>ArrowInterop.FromArrowTable</c> ignores this option. Default is false.
    /// </para>
    /// </remarks>
    public bool UseZeroCopy { get; set; } = false;

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
