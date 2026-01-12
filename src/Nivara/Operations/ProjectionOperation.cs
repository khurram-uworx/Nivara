using Nivara.Exceptions;
using Nivara.Query;

namespace Nivara.Operations;

/// <summary>
/// Represents a projection operation that selects and optionally renames columns
/// </summary>
internal sealed class ProjectionOperation : IQueryOperation
{
    readonly Dictionary<string, string?> columnMappings;

    /// <summary>
    /// Initializes a new instance of ProjectionOperation
    /// </summary>
    /// <param name="columnMappings">Dictionary mapping original column names to new names (null to keep original)</param>
    /// <exception cref="ArgumentNullException">Thrown when columnMappings is null</exception>
    /// <exception cref="ArgumentException">Thrown when no columns are specified</exception>
    public ProjectionOperation(Dictionary<string, string?> columnMappings)
    {
        this.columnMappings = columnMappings ?? throw new ArgumentNullException(nameof(columnMappings));

        if (columnMappings.Count == 0)
            throw new ArgumentException("Must specify at least one column mapping", nameof(columnMappings));

        // Validate that result column names are unique
        var resultNames = columnMappings.Values
            .Select(newName => newName ?? columnMappings.First(kvp => kvp.Value == newName).Key)
            .ToList();

        var duplicates = resultNames
            .GroupBy(name => name, StringComparer.OrdinalIgnoreCase)
            .Where(g => g.Count() > 1)
            .Select(g => g.Key)
            .ToList();

        if (duplicates.Any())
        {
            throw new ArgumentException($"Duplicate result column names found: {string.Join(", ", duplicates)}", nameof(columnMappings));
        }
    }

    /// <summary>
    /// Initializes a new instance of ProjectionOperation for simple column selection
    /// </summary>
    /// <param name="columnNames">The names of columns to select</param>
    /// <exception cref="ArgumentNullException">Thrown when columnNames is null</exception>
    /// <exception cref="ArgumentException">Thrown when no columns are specified</exception>
    public ProjectionOperation(IEnumerable<string> columnNames)
        : this(columnNames.ToDictionary(name => name, name => (string?)null))
    {
    }

    /// <inheritdoc />
    public string OperationType => "Projection";

    /// <inheritdoc />
    public Schema TransformSchema(Schema inputSchema)
    {
        if (inputSchema == null)
            throw new ArgumentNullException(nameof(inputSchema));

        // Validate that all source columns exist
        foreach (var originalName in columnMappings.Keys)
        {
            if (!inputSchema.HasColumn(originalName))
            {
                throw new SchemaValidationException(
                    $"Projection column '{originalName}' not found in schema. Available columns: {string.Join(", ", inputSchema.ColumnNames)}");
            }
        }

        // Create new schema with projected columns
        var projectedColumns = new List<(string Name, Type Type)>();
        foreach (var (originalName, newName) in columnMappings)
        {
            var finalName = newName ?? originalName;
            var columnType = inputSchema.GetColumnType(originalName);
            projectedColumns.Add((finalName, columnType));
        }

        return new Schema(projectedColumns);
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        try
        {
            var result = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);

            foreach (var (originalName, newName) in columnMappings)
            {
                if (!input.TryGetValue(originalName, out var column))
                {
                    throw new ColumnNotFoundException(originalName, input.Keys);
                }

                var finalName = newName ?? originalName;
                result[finalName] = column;
            }

            return result;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Projection operation failed: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Gets the column mappings for this projection
    /// </summary>
    public IReadOnlyDictionary<string, string?> ColumnMappings => columnMappings;

    /// <summary>
    /// Gets the source column names being projected
    /// </summary>
    public IReadOnlyList<string> SourceColumns => columnMappings.Keys.ToList();

    /// <summary>
    /// Gets the result column names after projection
    /// </summary>
    public IReadOnlyList<string> ResultColumns => columnMappings
        .Select(kvp => kvp.Value ?? kvp.Key)
        .ToList();

    /// <summary>
    /// Returns a string representation of the projection operation
    /// </summary>
    /// <returns>A string representation</returns>
    public override string ToString()
    {
        var mappings = columnMappings.Select(kvp =>
            kvp.Value == null ? kvp.Key : $"{kvp.Key} -> {kvp.Value}");
        return $"Project({string.Join(", ", mappings)})";
    }
}