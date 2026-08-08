using Nivara.Exceptions;
using Nivara.Expressions;
using Nivara.Operations;
using Nivara.Tensors;
using System.Collections.Concurrent;
using System.Linq.Expressions;
using System.Numerics;
using System.Numerics.Tensors;
using System.Reflection;

namespace Nivara;

/// <summary>
/// Extension methods for NivaraFrame to support transformations and projections
/// </summary>
public static partial class NivaraFrameExtensions
{
    /// <summary>
    /// Converts a typed column to a single-column NivaraFrame.
    /// </summary>
    /// <typeparam name="T">The column element type</typeparam>
    /// <param name="column">The typed column</param>
    /// <param name="name">The column name</param>
    /// <returns>A new NivaraFrame containing only this column</returns>
    /// <exception cref="ArgumentNullException">Thrown when column or name is null</exception>
    public static NivaraFrame ToFrame<T>(this NivaraColumn<T> column, string name)
        => NivaraFrame.Create(name, column);

    /// <summary>
    /// Creates a new frame with a transformed column
    /// </summary>
    /// <typeparam name="T">The type of the source column</typeparam>
    /// <typeparam name="TResult">The type of the result column</typeparam>
    /// <param name="frame">The source frame</param>
    /// <param name="columnName">The name of the column to transform</param>
    /// <param name="transform">The transformation function</param>
    /// <param name="resultColumnName">The name for the result column (defaults to original column name)</param>
    /// <returns>A new frame with the transformed column</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame or transform is null</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when the specified column is not found</exception>
    /// <exception cref="ColumnTypeMismatchException">Thrown when the column type doesn't match T</exception>
    public static NivaraFrame WithTransformedColumn<T, TResult>(
        this NivaraFrame frame,
        string columnName,
        Func<T, TResult> transform,
        string? resultColumnName = null)
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));
        if (transform == null)
            throw new ArgumentNullException(nameof(transform));

        // Get the source column
        var sourceColumn = frame.GetColumn<T>(columnName);

        // Transform the column
        var transformedColumn = sourceColumn.Transform(transform);

        // Determine result column name
        var finalResultColumnName = resultColumnName ?? columnName;

        // If replacing existing column, remove it first
        NivaraFrame resultFrame;
        if (string.Equals(columnName, finalResultColumnName, StringComparison.OrdinalIgnoreCase))
        {
            // Replacing existing column
            resultFrame = frame.WithoutColumn(columnName);
        }
        else
        {
            // Adding new column alongside existing
            resultFrame = frame;
        }

        // Add the transformed column
        return resultFrame.WithColumn(finalResultColumnName, transformedColumn);
    }

    /// <summary>
    /// Creates a new frame with multiple transformed columns
    /// </summary>
    /// <param name="frame">The source frame</param>
    /// <param name="transformations">Dictionary of column transformations (column name -> (transform function, result column name))</param>
    /// <returns>A new frame with the transformed columns</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame or transformations is null</exception>
    public static NivaraFrame WithTransformedColumns(
        this NivaraFrame frame,
        Dictionary<string, (Func<object, object> Transform, string? ResultColumnName)> transformations)
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));
        if (transformations == null)
            throw new ArgumentNullException(nameof(transformations));

        var resultFrame = frame;

        foreach (var (columnName, (transform, resultColumnName)) in transformations)
        {
            // Get the source column
            var sourceColumn = frame.GetColumn(columnName);

            // Apply transformation using reflection to handle different types
            var transformedColumn = TransformColumnGeneric(sourceColumn, transform);

            // Determine result column name
            var finalResultColumnName = resultColumnName ?? columnName;

            // If replacing existing column, remove it first
            if (string.Equals(columnName, finalResultColumnName, StringComparison.OrdinalIgnoreCase))
            {
                // Replacing existing column
                resultFrame = resultFrame.WithoutColumn(columnName);
            }

            // Add the transformed column
            resultFrame = resultFrame.WithColumn(finalResultColumnName, transformedColumn);
        }

        return resultFrame;
    }

    /// <summary>
    /// Creates a new frame with columns selected and optionally renamed
    /// </summary>
    /// <param name="frame">The source frame</param>
    /// <param name="columnSelections">Dictionary mapping original column names to new column names (null to keep original name)</param>
    /// <returns>A new frame with selected and renamed columns</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame or columnSelections is null</exception>
    /// <exception cref="ArgumentException">Thrown when no columns are selected</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when any specified column is not found</exception>
    public static NivaraFrame SelectAndRename(
        this NivaraFrame frame,
        Dictionary<string, string?> columnSelections)
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));
        if (columnSelections == null)
            throw new ArgumentNullException(nameof(columnSelections));
        if (columnSelections.Count == 0)
            throw new ArgumentException("Must specify at least one column selection", nameof(columnSelections));

        var selectedColumns = new List<(string Name, IColumn Column)>();

        foreach (var (originalName, newName) in columnSelections)
        {
            // Validate that the column exists
            if (!frame.HasColumn(originalName))
                throw new ColumnNotFoundException(originalName, frame.ColumnNames);

            var column = frame.GetColumn(originalName);
            var finalName = newName ?? originalName;

            selectedColumns.Add((finalName, column));
        }

        return new NivaraFrame(selectedColumns);
    }

    /// <summary>
    /// Creates a new frame with columns selected by names
    /// </summary>
    /// <param name="frame">The source frame</param>
    /// <param name="columnNames">The names of columns to select</param>
    /// <returns>A new frame with selected columns</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame or columnNames is null</exception>
    /// <exception cref="ArgumentException">Thrown when no columns are specified</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when any specified column is not found</exception>
    public static NivaraFrame Select(this NivaraFrame frame, params string[] columnNames)
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));
        if (columnNames == null)
            throw new ArgumentNullException(nameof(columnNames));

        return frame.SelectColumns(columnNames);
    }

    /// <summary>
    /// Creates a new frame with columns selected by names
    /// </summary>
    /// <param name="frame">The source frame</param>
    /// <param name="columnNames">The names of columns to select</param>
    /// <returns>A new frame with selected columns</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame or columnNames is null</exception>
    /// <exception cref="ArgumentException">Thrown when no columns are specified</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when any specified column is not found</exception>
    public static NivaraFrame Select(this NivaraFrame frame, IEnumerable<string> columnNames)
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));
        if (columnNames == null)
            throw new ArgumentNullException(nameof(columnNames));

        return frame.SelectColumns(columnNames);
    }

    /// <summary>
    /// Creates a new frame with a column renamed
    /// </summary>
    /// <param name="frame">The source frame</param>
    /// <param name="oldName">The current name of the column</param>
    /// <param name="newName">The new name for the column</param>
    /// <returns>A new frame with the column renamed</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame is null</exception>
    /// <exception cref="ArgumentException">Thrown when oldName or newName is null or whitespace</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when the column to rename is not found</exception>
    /// <exception cref="ArgumentException">Thrown when newName conflicts with an existing column</exception>
    public static NivaraFrame RenameColumn(this NivaraFrame frame, string oldName, string newName)
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));
        if (string.IsNullOrWhiteSpace(oldName))
            throw new ArgumentException("Old column name cannot be null or whitespace", nameof(oldName));
        if (string.IsNullOrWhiteSpace(newName))
            throw new ArgumentException("New column name cannot be null or whitespace", nameof(newName));

        if (!frame.HasColumn(oldName))
            throw new ColumnNotFoundException(oldName, frame.ColumnNames);

        if (frame.HasColumn(newName) && !string.Equals(oldName, newName, StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException($"Column '{newName}' already exists in the frame", nameof(newName));

        // If renaming to the same name (case-insensitive), return the original frame
        if (string.Equals(oldName, newName, StringComparison.OrdinalIgnoreCase))
            return frame;

        // Get the column to rename
        var column = frame.GetColumn(oldName);

        // Create new frame with all columns except the old one, then add with new name
        var resultFrame = frame.WithoutColumn(oldName);
        return resultFrame.WithColumn(newName, column);
    }

    /// <summary>
    /// Creates a new frame with multiple columns renamed
    /// </summary>
    /// <param name="frame">The source frame</param>
    /// <param name="columnRenames">Dictionary mapping old column names to new column names</param>
    /// <returns>A new frame with columns renamed</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame or columnRenames is null</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when any column to rename is not found</exception>
    /// <exception cref="ArgumentException">Thrown when any new name conflicts with existing columns</exception>
    public static NivaraFrame RenameColumns(this NivaraFrame frame, Dictionary<string, string> columnRenames)
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));
        if (columnRenames == null)
            throw new ArgumentNullException(nameof(columnRenames));

        if (columnRenames.Count == 0)
            return frame;

        // Validate all old column names exist
        foreach (var oldName in columnRenames.Keys)
        {
            if (!frame.HasColumn(oldName))
                throw new ColumnNotFoundException(oldName, frame.ColumnNames);
        }

        // Validate no conflicts in new names
        var newNames = columnRenames.Values.ToList();
        var existingNames = frame.ColumnNames.Except(columnRenames.Keys, StringComparer.OrdinalIgnoreCase).ToList();
        var conflicts = newNames.Intersect(existingNames, StringComparer.OrdinalIgnoreCase).ToList();

        if (conflicts.Any())
        {
            throw new ArgumentException($"New column names conflict with existing columns: {string.Join(", ", conflicts)}", nameof(columnRenames));
        }

        // Build the column mappings for projection
        var columnMappings = new Dictionary<string, string?>();

        // Add all columns, applying renames where specified
        foreach (var columnName in frame.ColumnNames)
        {
            if (columnRenames.TryGetValue(columnName, out var newName))
            {
                columnMappings[columnName] = newName;
            }
            else
            {
                columnMappings[columnName] = null; // Keep original name
            }
        }

        return frame.SelectAndRename(columnMappings);
    }

    /// <summary>
    /// Creates a new frame excluding specified columns
    /// </summary>
    /// <param name="frame">The source frame</param>
    /// <param name="columnsToExclude">The names of columns to exclude</param>
    /// <returns>A new frame without the specified columns</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame or columnsToExclude is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when trying to exclude all columns</exception>
    public static NivaraFrame Exclude(this NivaraFrame frame, params string[] columnsToExclude)
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));
        if (columnsToExclude == null)
            throw new ArgumentNullException(nameof(columnsToExclude));

        return frame.Exclude((IEnumerable<string>)columnsToExclude);
    }

    /// <summary>
    /// Creates a new frame excluding specified columns
    /// </summary>
    /// <param name="frame">The source frame</param>
    /// <param name="columnsToExclude">The names of columns to exclude</param>
    /// <returns>A new frame without the specified columns</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame or columnsToExclude is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when trying to exclude all columns</exception>
    public static NivaraFrame Exclude(this NivaraFrame frame, IEnumerable<string> columnsToExclude)
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));
        if (columnsToExclude == null)
            throw new ArgumentNullException(nameof(columnsToExclude));

        var excludeSet = new HashSet<string>(columnsToExclude, StringComparer.OrdinalIgnoreCase);
        var remainingColumns = frame.ColumnNames.Where(name => !excludeSet.Contains(name)).ToList();

        if (remainingColumns.Count == 0)
            throw new InvalidOperationException("Cannot exclude all columns from a frame. At least one column must remain.");

        return frame.SelectColumns(remainingColumns);
    }

    /// <summary>
    /// Creates a new frame with a computed column added
    /// </summary>
    /// <typeparam name="T1">The type of the first source column</typeparam>
    /// <typeparam name="TResult">The type of the result column</typeparam>
    /// <param name="frame">The source frame</param>
    /// <param name="sourceColumn1">The name of the first source column</param>
    /// <param name="computation">The computation function</param>
    /// <param name="resultColumnName">The name for the result column</param>
    /// <returns>A new frame with the computed column added</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame or computation is null</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when any source column is not found</exception>
    public static NivaraFrame WithComputedColumn<T1, TResult>(
        this NivaraFrame frame,
        string sourceColumn1,
        Func<T1, TResult> computation,
        string resultColumnName)
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));
        if (computation == null)
            throw new ArgumentNullException(nameof(computation));
        if (string.IsNullOrWhiteSpace(resultColumnName))
            throw new ArgumentException("Result column name cannot be null or whitespace", nameof(resultColumnName));

        var col1 = frame.GetColumn<T1>(sourceColumn1);
        var resultColumn = col1.Transform(computation);

        return frame.WithColumn(resultColumnName, resultColumn);
    }

    /// <summary>
    /// Creates a new frame with a computed column added using two source columns
    /// </summary>
    /// <typeparam name="T1">The type of the first source column</typeparam>
    /// <typeparam name="T2">The type of the second source column</typeparam>
    /// <typeparam name="TResult">The type of the result column</typeparam>
    /// <param name="frame">The source frame</param>
    /// <param name="sourceColumn1">The name of the first source column</param>
    /// <param name="sourceColumn2">The name of the second source column</param>
    /// <param name="computation">The computation function</param>
    /// <param name="resultColumnName">The name for the result column</param>
    /// <returns>A new frame with the computed column added</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame or computation is null</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when any source column is not found</exception>
    /// <exception cref="ArgumentException">Thrown when columns have different lengths</exception>
    public static NivaraFrame WithComputedColumn<T1, T2, TResult>(
        this NivaraFrame frame,
        string sourceColumn1,
        string sourceColumn2,
        Func<T1, T2, TResult> computation,
        string resultColumnName)
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));
        if (computation == null)
            throw new ArgumentNullException(nameof(computation));
        if (string.IsNullOrWhiteSpace(resultColumnName))
            throw new ArgumentException("Result column name cannot be null or whitespace", nameof(resultColumnName));

        var col1 = frame.GetColumn<T1>(sourceColumn1);
        var col2 = frame.GetColumn<T2>(sourceColumn2);

        if (col1.Length != col2.Length)
            throw new ArgumentException($"Source columns have different lengths: {sourceColumn1}({col1.Length}) vs {sourceColumn2}({col2.Length})");

        // Create result column by combining values from both source columns
        var result = new TResult[col1.Length];
        var resultNullMask = new bool[col1.Length];
        bool hasResultNulls = false;

        for (int i = 0; i < col1.Length; i++)
        {
            bool col1IsNull = col1.IsNull(i);
            bool col2IsNull = col2.IsNull(i);

            if (col1IsNull || col2IsNull)
            {
                // If either input is null, result is null
                result[i] = default(TResult)!;
                resultNullMask[i] = true;
                hasResultNulls = true;
            }
            else
            {
                try
                {
                    var computedValue = computation(col1[i], col2[i]);

                    // Check if the computed value is null for reference types
                    if (!typeof(TResult).IsValueType && computedValue == null)
                    {
                        result[i] = default(TResult)!;
                        resultNullMask[i] = true;
                        hasResultNulls = true;
                    }
                    else
                    {
                        result[i] = computedValue;
                        resultNullMask[i] = false;
                    }
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException(
                        $"Computation function threw an exception at index {i}. " +
                        $"Input values: {sourceColumn1}={col1[i]}, {sourceColumn2}={col2[i]}, Exception: {ex.Message}", ex);
                }
            }
        }

        // Create result column with appropriate null handling
        IColumn resultColumn;
        if (typeof(TResult).IsValueType)
        {
            if (hasResultNulls)
            {
                var nullableType = typeof(Nullable<>).MakeGenericType(typeof(TResult));
                var nullableArray = System.Array.CreateInstance(nullableType, result.Length);

                for (int i = 0; i < result.Length; i++)
                {
                    if (resultNullMask[i])
                    {
                        nullableArray.SetValue(null, i);
                    }
                    else
                    {
                        var nullableInstance = Activator.CreateInstance(nullableType, result[i]);
                        nullableArray.SetValue(nullableInstance, i);
                    }
                }

                resultColumn = (IColumn)typeof(NivaraColumn<>)
                    .MakeGenericType(typeof(TResult))
                    .GetMethod(nameof(NivaraColumn<int>.CreateFromNullable), new[] { nullableType.MakeArrayType() })!
                    .Invoke(null, new object[] { nullableArray })!;
            }
            else
            {
                resultColumn = NivaraColumn<TResult>.Create(result);
            }
        }
        else
        {
            resultColumn = NivaraColumn<TResult>.CreateForReferenceType(result);
        }

        return frame.WithColumn(resultColumnName, resultColumn);
    }

    /// <summary>
    /// Helper method to transform a column using object-based transformation (for generic scenarios)
    /// </summary>
    /// <param name="sourceColumn">The source column</param>
    /// <param name="transform">The transformation function</param>
    /// <returns>The transformed column</returns>
    private static IColumn TransformColumnGeneric(IColumn sourceColumn, Func<object, object> transform)
    {
        var elementType = sourceColumn.ElementType;

        // Use reflection to call the appropriate Transform method
        var transformMethod = typeof(NivaraColumn<>)
            .MakeGenericType(elementType)
            .GetMethod("Transform");

        if (transformMethod == null)
            throw new InvalidOperationException($"Transform method not found for type {elementType.Name}");

        // Create a typed transformation function
        var typedTransform = CreateTypedTransform(transform, elementType);

        return (IColumn)transformMethod.Invoke(sourceColumn, new[] { typedTransform })!;
    }

    /// <summary>
    /// Creates a typed transformation function from an object-based one
    /// </summary>
    /// <param name="objectTransform">The object-based transformation function</param>
    /// <param name="sourceType">The source type</param>
    /// <returns>A typed transformation function</returns>
    private static object CreateTypedTransform(Func<object, object> objectTransform, Type sourceType)
    {
        // Create a delegate of type Func<T, object> where T is the source type
        var delegateType = typeof(Func<,>).MakeGenericType(sourceType, typeof(object));

        // Create the typed wrapper function
        var method = typeof(NivaraFrameExtensions)
            .GetMethod(nameof(TypedTransformWrapper), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!
            .MakeGenericMethod(sourceType);

        return Delegate.CreateDelegate(delegateType, objectTransform, method);
    }

    /// <summary>
    /// Wrapper method to convert typed input to object for transformation
    /// </summary>
    /// <typeparam name="T">The input type</typeparam>
    /// <param name="transform">The object-based transformation function</param>
    /// <param name="input">The typed input value</param>
    /// <returns>The transformed value</returns>
    private static object TypedTransformWrapper<T>(Func<object, object> transform, T input)
    {
        return transform(input!);
    }

    #region Join Operations

    /// <summary>
    /// Performs an inner join with another DataFrame
    /// </summary>
    /// <param name="left">The left DataFrame</param>
    /// <param name="right">The right DataFrame</param>
    /// <param name="joinKey">The join key (same column name in both DataFrames)</param>
    /// <param name="disambiguationStrategy">Strategy for handling column name conflicts</param>
    /// <param name="leftPrefix">Prefix for left columns when using prefix disambiguation</param>
    /// <param name="rightPrefix">Prefix for right columns when using prefix disambiguation</param>
    /// <returns>A new DataFrame containing the inner join result</returns>
    /// <exception cref="ArgumentNullException">Thrown when any parameter is null</exception>
    /// <exception cref="SchemaValidationException">Thrown when join keys are incompatible</exception>
    public static NivaraFrame InnerJoin(
        this NivaraFrame left,
        NivaraFrame right,
        string joinKey,
        ColumnDisambiguationStrategy disambiguationStrategy = ColumnDisambiguationStrategy.Suffix,
        string leftPrefix = "left",
        string rightPrefix = "right")
    {
        if (left == null)
            throw new ArgumentNullException(nameof(left));
        if (right == null)
            throw new ArgumentNullException(nameof(right));

        return left.Join(right, JoinType.Inner, new[] { new JoinKey(joinKey) }, disambiguationStrategy, leftPrefix, rightPrefix);
    }

    /// <summary>
    /// Performs an inner join with another DataFrame using different column names
    /// </summary>
    /// <param name="left">The left DataFrame</param>
    /// <param name="right">The right DataFrame</param>
    /// <param name="leftKey">The join key column name in the left DataFrame</param>
    /// <param name="rightKey">The join key column name in the right DataFrame</param>
    /// <param name="disambiguationStrategy">Strategy for handling column name conflicts</param>
    /// <param name="leftPrefix">Prefix for left columns when using prefix disambiguation</param>
    /// <param name="rightPrefix">Prefix for right columns when using prefix disambiguation</param>
    /// <returns>A new DataFrame containing the inner join result</returns>
    /// <exception cref="ArgumentNullException">Thrown when any parameter is null</exception>
    /// <exception cref="SchemaValidationException">Thrown when join keys are incompatible</exception>
    public static NivaraFrame InnerJoin(
        this NivaraFrame left,
        NivaraFrame right,
        string leftKey,
        string rightKey,
        ColumnDisambiguationStrategy disambiguationStrategy = ColumnDisambiguationStrategy.Suffix,
        string leftPrefix = "left",
        string rightPrefix = "right")
    {
        if (left == null)
            throw new ArgumentNullException(nameof(left));
        if (right == null)
            throw new ArgumentNullException(nameof(right));

        return left.Join(right, JoinType.Inner, new[] { new JoinKey(leftKey, rightKey) }, disambiguationStrategy, leftPrefix, rightPrefix);
    }

    /// <summary>
    /// Performs a left join with another DataFrame
    /// </summary>
    /// <param name="left">The left DataFrame</param>
    /// <param name="right">The right DataFrame</param>
    /// <param name="joinKey">The join key (same column name in both DataFrames)</param>
    /// <param name="disambiguationStrategy">Strategy for handling column name conflicts</param>
    /// <param name="leftPrefix">Prefix for left columns when using prefix disambiguation</param>
    /// <param name="rightPrefix">Prefix for right columns when using prefix disambiguation</param>
    /// <returns>A new DataFrame containing the left join result</returns>
    /// <exception cref="ArgumentNullException">Thrown when any parameter is null</exception>
    /// <exception cref="SchemaValidationException">Thrown when join keys are incompatible</exception>
    public static NivaraFrame LeftJoin(
        this NivaraFrame left,
        NivaraFrame right,
        string joinKey,
        ColumnDisambiguationStrategy disambiguationStrategy = ColumnDisambiguationStrategy.Suffix,
        string leftPrefix = "left",
        string rightPrefix = "right")
    {
        if (left == null)
            throw new ArgumentNullException(nameof(left));
        if (right == null)
            throw new ArgumentNullException(nameof(right));

        return left.Join(right, JoinType.Left, new[] { new JoinKey(joinKey) }, disambiguationStrategy, leftPrefix, rightPrefix);
    }

    /// <summary>
    /// Performs a left join with another DataFrame using different column names
    /// </summary>
    /// <param name="left">The left DataFrame</param>
    /// <param name="right">The right DataFrame</param>
    /// <param name="leftKey">The join key column name in the left DataFrame</param>
    /// <param name="rightKey">The join key column name in the right DataFrame</param>
    /// <param name="disambiguationStrategy">Strategy for handling column name conflicts</param>
    /// <param name="leftPrefix">Prefix for left columns when using prefix disambiguation</param>
    /// <param name="rightPrefix">Prefix for right columns when using prefix disambiguation</param>
    /// <returns>A new DataFrame containing the left join result</returns>
    /// <exception cref="ArgumentNullException">Thrown when any parameter is null</exception>
    /// <exception cref="SchemaValidationException">Thrown when join keys are incompatible</exception>
    public static NivaraFrame LeftJoin(
        this NivaraFrame left,
        NivaraFrame right,
        string leftKey,
        string rightKey,
        ColumnDisambiguationStrategy disambiguationStrategy = ColumnDisambiguationStrategy.Suffix,
        string leftPrefix = "left",
        string rightPrefix = "right")
    {
        if (left == null)
            throw new ArgumentNullException(nameof(left));
        if (right == null)
            throw new ArgumentNullException(nameof(right));

        return left.Join(right, JoinType.Left, new[] { new JoinKey(leftKey, rightKey) }, disambiguationStrategy, leftPrefix, rightPrefix);
    }

    /// <summary>
    /// Performs a right join with another DataFrame
    /// </summary>
    /// <param name="left">The left DataFrame</param>
    /// <param name="right">The right DataFrame</param>
    /// <param name="joinKey">The join key (same column name in both DataFrames)</param>
    /// <param name="disambiguationStrategy">Strategy for handling column name conflicts</param>
    /// <param name="leftPrefix">Prefix for left columns when using prefix disambiguation</param>
    /// <param name="rightPrefix">Prefix for right columns when using prefix disambiguation</param>
    /// <returns>A new DataFrame containing the right join result</returns>
    /// <exception cref="ArgumentNullException">Thrown when any parameter is null</exception>
    /// <exception cref="SchemaValidationException">Thrown when join keys are incompatible</exception>
    public static NivaraFrame RightJoin(
        this NivaraFrame left,
        NivaraFrame right,
        string joinKey,
        ColumnDisambiguationStrategy disambiguationStrategy = ColumnDisambiguationStrategy.Suffix,
        string leftPrefix = "left",
        string rightPrefix = "right")
    {
        if (left == null)
            throw new ArgumentNullException(nameof(left));
        if (right == null)
            throw new ArgumentNullException(nameof(right));

        return left.Join(right, JoinType.Right, new[] { new JoinKey(joinKey) }, disambiguationStrategy, leftPrefix, rightPrefix);
    }

    /// <summary>
    /// Performs a right join with another DataFrame using different column names
    /// </summary>
    /// <param name="left">The left DataFrame</param>
    /// <param name="right">The right DataFrame</param>
    /// <param name="leftKey">The join key column name in the left DataFrame</param>
    /// <param name="rightKey">The join key column name in the right DataFrame</param>
    /// <param name="disambiguationStrategy">Strategy for handling column name conflicts</param>
    /// <param name="leftPrefix">Prefix for left columns when using prefix disambiguation</param>
    /// <param name="rightPrefix">Prefix for right columns when using prefix disambiguation</param>
    /// <returns>A new DataFrame containing the right join result</returns>
    /// <exception cref="ArgumentNullException">Thrown when any parameter is null</exception>
    /// <exception cref="SchemaValidationException">Thrown when join keys are incompatible</exception>
    public static NivaraFrame RightJoin(
        this NivaraFrame left,
        NivaraFrame right,
        string leftKey,
        string rightKey,
        ColumnDisambiguationStrategy disambiguationStrategy = ColumnDisambiguationStrategy.Suffix,
        string leftPrefix = "left",
        string rightPrefix = "right")
    {
        if (left == null)
            throw new ArgumentNullException(nameof(left));
        if (right == null)
            throw new ArgumentNullException(nameof(right));

        return left.Join(right, JoinType.Right, new[] { new JoinKey(leftKey, rightKey) }, disambiguationStrategy, leftPrefix, rightPrefix);
    }

    /// <summary>
    /// Performs a full outer join with another DataFrame
    /// </summary>
    /// <param name="left">The left DataFrame</param>
    /// <param name="right">The right DataFrame</param>
    /// <param name="joinKey">The join key (same column name in both DataFrames)</param>
    /// <param name="disambiguationStrategy">Strategy for handling column name conflicts</param>
    /// <param name="leftPrefix">Prefix for left columns when using prefix disambiguation</param>
    /// <param name="rightPrefix">Prefix for right columns when using prefix disambiguation</param>
    /// <returns>A new DataFrame containing the full outer join result</returns>
    /// <exception cref="ArgumentNullException">Thrown when any parameter is null</exception>
    /// <exception cref="SchemaValidationException">Thrown when join keys are incompatible</exception>
    public static NivaraFrame FullOuterJoin(
        this NivaraFrame left,
        NivaraFrame right,
        string joinKey,
        ColumnDisambiguationStrategy disambiguationStrategy = ColumnDisambiguationStrategy.Suffix,
        string leftPrefix = "left",
        string rightPrefix = "right")
    {
        if (left == null)
            throw new ArgumentNullException(nameof(left));
        if (right == null)
            throw new ArgumentNullException(nameof(right));

        return left.Join(right, JoinType.FullOuter, new[] { new JoinKey(joinKey) }, disambiguationStrategy, leftPrefix, rightPrefix);
    }

    /// <summary>
    /// Performs a full outer join with another DataFrame using different column names
    /// </summary>
    /// <param name="left">The left DataFrame</param>
    /// <param name="right">The right DataFrame</param>
    /// <param name="leftKey">The join key column name in the left DataFrame</param>
    /// <param name="rightKey">The join key column name in the right DataFrame</param>
    /// <param name="disambiguationStrategy">Strategy for handling column name conflicts</param>
    /// <param name="leftPrefix">Prefix for left columns when using prefix disambiguation</param>
    /// <param name="rightPrefix">Prefix for right columns when using prefix disambiguation</param>
    /// <returns>A new DataFrame containing the full outer join result</returns>
    /// <exception cref="ArgumentNullException">Thrown when any parameter is null</exception>
    /// <exception cref="SchemaValidationException">Thrown when join keys are incompatible</exception>
    public static NivaraFrame FullOuterJoin(
        this NivaraFrame left,
        NivaraFrame right,
        string leftKey,
        string rightKey,
        ColumnDisambiguationStrategy disambiguationStrategy = ColumnDisambiguationStrategy.Suffix,
        string leftPrefix = "left",
        string rightPrefix = "right")
    {
        if (left == null)
            throw new ArgumentNullException(nameof(left));
        if (right == null)
            throw new ArgumentNullException(nameof(right));

        return left.Join(right, JoinType.FullOuter, new[] { new JoinKey(leftKey, rightKey) }, disambiguationStrategy, leftPrefix, rightPrefix);
    }

    /// <summary>
    /// Performs a join with another DataFrame using multiple join keys
    /// </summary>
    /// <param name="left">The left DataFrame</param>
    /// <param name="right">The right DataFrame</param>
    /// <param name="joinType">The type of join to perform</param>
    /// <param name="joinKeys">The join keys</param>
    /// <param name="disambiguationStrategy">Strategy for handling column name conflicts</param>
    /// <param name="leftPrefix">Prefix for left columns when using prefix disambiguation</param>
    /// <param name="rightPrefix">Prefix for right columns when using prefix disambiguation</param>
    /// <returns>A new DataFrame containing the join result</returns>
    /// <exception cref="ArgumentNullException">Thrown when any parameter is null</exception>
    /// <exception cref="SchemaValidationException">Thrown when join keys are incompatible</exception>
    public static NivaraFrame Join(
        this NivaraFrame left,
        NivaraFrame right,
        JoinType joinType,
        JoinKey[] joinKeys,
        ColumnDisambiguationStrategy disambiguationStrategy = ColumnDisambiguationStrategy.Suffix,
        string leftPrefix = "left",
        string rightPrefix = "right")
    {
        if (left == null)
            throw new ArgumentNullException(nameof(left));
        if (right == null)
            throw new ArgumentNullException(nameof(right));
        if (joinKeys == null)
            throw new ArgumentNullException(nameof(joinKeys));

        // Convert frames to column dictionaries
        var leftColumns = left.ColumnNames.ToDictionary(name => name, name => left.GetColumn(name), StringComparer.OrdinalIgnoreCase);
        var rightColumns = right.ColumnNames.ToDictionary(name => name, name => right.GetColumn(name), StringComparer.OrdinalIgnoreCase);

        // Create and execute join operation
        var joinOperation = new JoinOperation(
            leftColumns,
            rightColumns,
            joinType,
            joinKeys,
            disambiguationStrategy,
            leftPrefix,
            rightPrefix);

        // Execute the join (note: input parameter is not used for join operations)
        var resultColumns = joinOperation.Execute(leftColumns);

        // Convert result back to NivaraFrame
        var namedColumns = resultColumns.Select(kvp => (kvp.Key, kvp.Value));
        return new NivaraFrame(namedColumns);
    }

    #endregion

    #region Concatenation Operations

    /// <summary>
    /// Concatenates multiple DataFrames vertically (appending rows)
    /// </summary>
    /// <param name="frames">The DataFrames to concatenate</param>
    /// <param name="mismatchHandling">How to handle schema mismatches</param>
    /// <returns>A new DataFrame containing all rows from the input DataFrames</returns>
    /// <exception cref="ArgumentNullException">Thrown when frames is null</exception>
    /// <exception cref="ArgumentException">Thrown when no frames are provided</exception>
    /// <exception cref="SchemaValidationException">Thrown when schemas are incompatible and mismatchHandling is Error</exception>
    public static NivaraFrame ConcatenateVertical(
        IEnumerable<NivaraFrame> frames,
        ConcatenationMismatchHandling mismatchHandling = ConcatenationMismatchHandling.FillWithNulls)
    {
        if (frames == null)
            throw new ArgumentNullException(nameof(frames));

        var frameList = frames.ToList();
        if (frameList.Count == 0)
            throw new ArgumentException("Must provide at least one frame for concatenation", nameof(frames));

        if (frameList.Count == 1)
            return frameList[0];

        // Convert frames to column dictionaries
        var columnDictionaries = frameList.Select(frame =>
            frame.ColumnNames.ToDictionary(name => name, name => frame.GetColumn(name), StringComparer.OrdinalIgnoreCase)
        ).ToList();

        var concatenationOperation = new ConcatenationOperation(
            columnDictionaries.Skip(1).ToList(),
            ConcatenationDirection.Vertical,
            mismatchHandling);

        // Execute the concatenation
        var resultColumns = concatenationOperation.Execute(columnDictionaries[0]);

        // Convert result back to NivaraFrame
        var namedColumns = resultColumns.Select(kvp => (kvp.Key, kvp.Value));
        return new NivaraFrame(namedColumns);
    }

    /// <summary>
    /// Concatenates this DataFrame with another DataFrame vertically (appending rows)
    /// </summary>
    /// <param name="first">The first DataFrame</param>
    /// <param name="second">The second DataFrame to append</param>
    /// <param name="mismatchHandling">How to handle schema mismatches</param>
    /// <returns>A new DataFrame containing rows from both DataFrames</returns>
    /// <exception cref="ArgumentNullException">Thrown when any parameter is null</exception>
    /// <exception cref="SchemaValidationException">Thrown when schemas are incompatible and mismatchHandling is Error</exception>
    public static NivaraFrame ConcatenateVertical(
        this NivaraFrame first,
        NivaraFrame second,
        ConcatenationMismatchHandling mismatchHandling = ConcatenationMismatchHandling.FillWithNulls)
    {
        if (first == null)
            throw new ArgumentNullException(nameof(first));
        if (second == null)
            throw new ArgumentNullException(nameof(second));

        return ConcatenateVertical(new[] { first, second }, mismatchHandling);
    }

    /// <summary>
    /// Concatenates multiple DataFrames horizontally (appending columns)
    /// </summary>
    /// <param name="frames">The DataFrames to concatenate</param>
    /// <returns>A new DataFrame containing all columns from the input DataFrames</returns>
    /// <exception cref="ArgumentNullException">Thrown when frames is null</exception>
    /// <exception cref="ArgumentException">Thrown when no frames are provided or row counts don't match</exception>
    /// <exception cref="SchemaValidationException">Thrown when column names conflict</exception>
    public static NivaraFrame ConcatenateHorizontal(IEnumerable<NivaraFrame> frames)
    {
        if (frames == null)
            throw new ArgumentNullException(nameof(frames));

        var frameList = frames.ToList();
        if (frameList.Count == 0)
            throw new ArgumentException("Must provide at least one frame for concatenation", nameof(frames));

        if (frameList.Count == 1)
            return frameList[0];

        // Convert frames to column dictionaries
        var columnDictionaries = frameList.Select(frame =>
            frame.ColumnNames.ToDictionary(name => name, name => frame.GetColumn(name), StringComparer.OrdinalIgnoreCase)
        ).ToList();

        var concatenationOperation = new ConcatenationOperation(
            columnDictionaries.Skip(1).ToList(),
            ConcatenationDirection.Horizontal,
            ConcatenationMismatchHandling.Error); // Horizontal concatenation always uses Error for mismatches

        // Execute the concatenation
        var resultColumns = concatenationOperation.Execute(columnDictionaries[0]);

        // Convert result back to NivaraFrame
        var namedColumns = resultColumns.Select(kvp => (kvp.Key, kvp.Value));
        return new NivaraFrame(namedColumns);
    }

    /// <summary>
    /// Concatenates this DataFrame with another DataFrame horizontally (appending columns)
    /// </summary>
    /// <param name="first">The first DataFrame</param>
    /// <param name="second">The second DataFrame to append columns from</param>
    /// <returns>A new DataFrame containing columns from both DataFrames</returns>
    /// <exception cref="ArgumentNullException">Thrown when any parameter is null</exception>
    /// <exception cref="ArgumentException">Thrown when row counts don't match</exception>
    /// <exception cref="SchemaValidationException">Thrown when column names conflict</exception>
    public static NivaraFrame ConcatenateHorizontal(this NivaraFrame first, NivaraFrame second)
    {
        if (first == null)
            throw new ArgumentNullException(nameof(first));
        if (second == null)
            throw new ArgumentNullException(nameof(second));

        return ConcatenateHorizontal(new[] { first, second });
    }

    /// <summary>
    /// Appends rows from another DataFrame to this DataFrame (alias for ConcatenateVertical)
    /// </summary>
    /// <param name="first">The first DataFrame</param>
    /// <param name="second">The DataFrame to append</param>
    /// <param name="mismatchHandling">How to handle schema mismatches</param>
    /// <returns>A new DataFrame with appended rows</returns>
    /// <exception cref="ArgumentNullException">Thrown when any parameter is null</exception>
    /// <exception cref="SchemaValidationException">Thrown when schemas are incompatible and mismatchHandling is Error</exception>
    public static NivaraFrame Append(
        this NivaraFrame first,
        NivaraFrame second,
        ConcatenationMismatchHandling mismatchHandling = ConcatenationMismatchHandling.FillWithNulls)
    {
        return first.ConcatenateVertical(second, mismatchHandling);
    }

    /// <summary>
    /// Combines columns from another DataFrame with this DataFrame (alias for ConcatenateHorizontal)
    /// </summary>
    /// <param name="first">The first DataFrame</param>
    /// <param name="second">The DataFrame to combine columns from</param>
    /// <returns>A new DataFrame with combined columns</returns>
    /// <exception cref="ArgumentNullException">Thrown when any parameter is null</exception>
    /// <exception cref="ArgumentException">Thrown when row counts don't match</exception>
    /// <exception cref="SchemaValidationException">Thrown when column names conflict</exception>
    public static NivaraFrame Combine(this NivaraFrame first, NivaraFrame second)
    {
        return first.ConcatenateHorizontal(second);
    }

    #endregion

    #region Fluent API Operations

    /// <summary>
    /// Filters rows based on a predicate over a typed <see cref="NivaraRow"/>
    /// </summary>
    /// <param name="frame">The source DataFrame</param>
    /// <param name="predicate">The predicate expression to filter by</param>
    /// <returns>A new DataFrame containing only rows where the predicate is true</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame or predicate is null</exception>
    public static NivaraFrame Where(this NivaraFrame frame, Func<NivaraRow, bool> predicate)
    {
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentNullException.ThrowIfNull(predicate);

        var columns = frame.ColumnNames.Select(name => frame.GetColumn(name)).ToArray();
        var map = new Dictionary<string, int>(frame.ColumnNames.Count, StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < frame.ColumnNames.Count; i++)
            map[frame.ColumnNames[i]] = i;

        var mask = new bool[frame.RowCount];
        for (int i = 0; i < frame.RowCount; i++)
            mask[i] = predicate(new NivaraRow(columns, map, i));

        return frame.FilterByMask(NivaraColumn<bool>.Create(mask));
    }

    /// <summary>
    /// Sorts the DataFrame by a single column
    /// </summary>
    /// <param name="frame">The source DataFrame</param>
    /// <param name="columnName">The name of the column to sort by</param>
    /// <param name="ascending">Whether to sort in ascending order (default: true)</param>
    /// <returns>A new DataFrame sorted by the specified column</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame is null</exception>
    /// <exception cref="ArgumentException">Thrown when columnName is null or whitespace</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when the column is not found</exception>
    public static NivaraFrame OrderBy(this NivaraFrame frame, string columnName, bool ascending = true)
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));
        if (string.IsNullOrWhiteSpace(columnName))
            throw new ArgumentException("Column name cannot be null or whitespace", nameof(columnName));

        var direction = ascending ? SortDirection.Ascending : SortDirection.Descending;
        var sortKey = new SortKey(columnName, direction, NullOrdering.NullsLast);

        return frame.Sort(new[] { sortKey });
    }

    /// <summary>
    /// Sorts the DataFrame by multiple columns
    /// </summary>
    /// <param name="frame">The source DataFrame</param>
    /// <param name="sortKeys">The sort keys defining the sort order and priority</param>
    /// <returns>A new DataFrame sorted by the specified columns</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame or sortKeys is null</exception>
    /// <exception cref="ArgumentException">Thrown when no sort keys are provided</exception>
    public static NivaraFrame OrderBy(this NivaraFrame frame, params SortKey[] sortKeys)
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));
        if (sortKeys == null)
            throw new ArgumentNullException(nameof(sortKeys));
        if (sortKeys.Length == 0)
            throw new ArgumentException("Must provide at least one sort key", nameof(sortKeys));

        return frame.Sort(sortKeys);
    }

    /// <summary>
    /// Applies a stable secondary sort by a column, extending the primary sort from a preceding
    /// <see cref="OrderBy(NivaraFrame, string, bool)"/> so the ordering composes lexicographically
    /// (primary key first, then this secondary key). Without a preceding sort, acts as a primary sort.
    /// </summary>
    /// <param name="frame">The source DataFrame</param>
    /// <param name="columnName">The name of the column to sort by</param>
    /// <param name="ascending">Whether to sort in ascending order (default: true)</param>
    /// <returns>A new DataFrame sorted by the specified column</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame is null</exception>
    /// <exception cref="ArgumentException">Thrown when columnName is null or whitespace</exception>
    public static NivaraFrame ThenBy(this NivaraFrame frame, string columnName, bool ascending = true)
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));
        if (string.IsNullOrWhiteSpace(columnName))
            throw new ArgumentException("Column name cannot be null or whitespace", nameof(columnName));

        var direction = ascending ? SortDirection.Ascending : SortDirection.Descending;
        var newKey = new SortKey(columnName, direction, NullOrdering.NullsLast);

        var mergedKeys = frame.SortKeys is { } existing
            ? existing.Concat(new[] { newKey }).ToArray()
            : new[] { newKey };

        return frame.Sort(mergedKeys);
    }

    /// <summary>
    /// Applies a stable secondary descending sort by a column, extending the primary sort from a
    /// preceding <see cref="OrderBy(NivaraFrame, string, bool)"/>. Without a preceding sort, acts as a
    /// primary descending sort.
    /// </summary>
    /// <param name="frame">The source DataFrame</param>
    /// <param name="columnName">The name of the column to sort by</param>
    /// <returns>A new DataFrame sorted by the specified column</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame is null</exception>
    /// <exception cref="ArgumentException">Thrown when columnName is null or whitespace</exception>
    public static NivaraFrame ThenByDescending(this NivaraFrame frame, string columnName)
    {
        return frame.ThenBy(columnName, ascending: false);
    }

    /// <summary>
    /// Groups the DataFrame by the specified columns
    /// </summary>
    /// <param name="frame">The source DataFrame</param>
    /// <param name="keyColumns">The names of the columns to group by</param>
    /// <returns>A new DataFrame with grouped data</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame or keyColumns is null</exception>
    /// <exception cref="ArgumentException">Thrown when no key columns are provided</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when any key column is not found</exception>
    public static NivaraFrame GroupBy(this NivaraFrame frame, params string[] keyColumns)
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));
        if (keyColumns == null)
            throw new ArgumentNullException(nameof(keyColumns));
        if (keyColumns.Length == 0)
            throw new ArgumentException("Must provide at least one key column", nameof(keyColumns));

        // Validate that all key columns exist
        foreach (var columnName in keyColumns)
        {
            if (!frame.HasColumn(columnName))
                throw new ColumnNotFoundException(columnName, frame.ColumnNames);
        }

        // Convert column names to ColumnExpressions
        var columnExpressions = keyColumns.Select(name => ColumnExpressions.Col(name)).ToArray();

        // Convert frame to column dictionary
        var columns = frame.ColumnNames.ToDictionary(name => name, name => frame.GetColumn(name), StringComparer.OrdinalIgnoreCase);

        // Create and execute group by operation
        var groupByOperation = new GroupByOperation(columnExpressions);
        var resultColumns = groupByOperation.Execute(columns);

        // Convert result back to NivaraFrame
        var namedColumns = resultColumns.Select(kvp => (kvp.Key, kvp.Value));
        return new NivaraFrame(namedColumns);
    }

    /// <summary>
    /// Groups the DataFrame by the specified columns and applies aggregations
    /// </summary>
    /// <param name="frame">The source DataFrame</param>
    /// <param name="keyColumns">The names of the columns to group by</param>
    /// <param name="aggregations">Dictionary mapping column names to aggregation functions</param>
    /// <returns>A new DataFrame with grouped and aggregated data</returns>
    /// <exception cref="ArgumentNullException">Thrown when any parameter is null</exception>
    /// <exception cref="ArgumentException">Thrown when no key columns are provided</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when any column is not found</exception>
    public static NivaraFrame GroupBy(this NivaraFrame frame, string[] keyColumns, Dictionary<string, AggregationFunction> aggregations)
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));
        if (keyColumns == null)
            throw new ArgumentNullException(nameof(keyColumns));
        if (aggregations == null)
            throw new ArgumentNullException(nameof(aggregations));
        if (keyColumns.Length == 0)
            throw new ArgumentException("Must provide at least one key column", nameof(keyColumns));

        // Validate that all columns exist
        foreach (var columnName in keyColumns.Concat(aggregations.Keys))
        {
            if (!frame.HasColumn(columnName))
                throw new ColumnNotFoundException(columnName, frame.ColumnNames);
        }

        // Convert column names to ColumnExpressions
        var columnExpressions = keyColumns.Select(name => ColumnExpressions.Col(name)).ToArray();

        // Convert frame to column dictionary
        var columns = frame.ColumnNames.ToDictionary(name => name, name => frame.GetColumn(name), StringComparer.OrdinalIgnoreCase);

        // Create and execute group by operation
        var groupByOperation = new GroupByOperation(columnExpressions);
        var resultColumns = groupByOperation.Execute(columns);

        // Apply aggregations to the grouped data
        // Note: This is a simplified implementation. In a full implementation,
        // we would need to modify GroupByOperation to support aggregations directly
        // For now, we just return the grouped keys

        // Convert result back to NivaraFrame
        var namedColumns = resultColumns.Select(kvp => (kvp.Key, kvp.Value));
        return new NivaraFrame(namedColumns);
    }

    /// <summary>
    /// Executes any pending operations and returns a materialized DataFrame (execution barrier)
    /// For already materialized DataFrames, this returns the frame as-is
    /// </summary>
    /// <param name="frame">The source DataFrame</param>
    /// <returns>A materialized DataFrame</returns>
    /// <exception cref="ArgumentNullException">Thrown when frame is null</exception>
    public static NivaraFrame Collect(this NivaraFrame frame)
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));

        // For materialized frames, Collect() is a no-op
        return frame;
    }

    /// <summary>
    /// Helper method to sort a DataFrame using SortOperation
    /// </summary>
    /// <param name="frame">The source DataFrame</param>
    /// <param name="sortKeys">The sort keys</param>
    /// <returns>A new sorted DataFrame</returns>
    private static NivaraFrame Sort(this NivaraFrame frame, SortKey[] sortKeys)
    {
        // Convert frame to column dictionary
        var columns = frame.ColumnNames.ToDictionary(name => name, name => frame.GetColumn(name), StringComparer.OrdinalIgnoreCase);

        // Create and execute sort operation
        var sortOperation = new SortOperation(sortKeys, stable: true);
        var resultColumns = sortOperation.Execute(columns);

        // Convert result back to NivaraFrame
        var namedColumns = resultColumns.Select(kvp => (kvp.Key, kvp.Value));
        var result = new NivaraFrame(namedColumns);
        result.SortKeys = sortKeys;
        return result;
    }

    #endregion

    #region Data Preparation

    /// <summary>
    /// Standardizes (z-score normalizes) numeric columns in a NivaraFrame to zero mean and unit variance.
    /// This is an alias for <see cref="Normalize(NivaraFrame, string[])"/>.
    /// </summary>
    /// <param name="frame">The source NivaraFrame</param>
    /// <param name="columns">The columns to standardize (null for all float/double numeric columns)</param>
    /// <returns>A new NivaraFrame with standardized columns</returns>
    public static NivaraFrame Standardize(this NivaraFrame frame, params string[]? columns)
        => Normalize(frame, columns);

    /// <summary>
    /// Normalizes numeric columns in a NivaraFrame to zero mean and unit variance (z-score).
    /// Null values are skipped when computing statistics and remain null in the result.
    /// </summary>
    /// <param name="frame">The source NivaraFrame</param>
    /// <param name="columns">The columns to normalize (null for all float/double numeric columns)</param>
    /// <returns>A new NivaraFrame with normalized columns</returns>
    public static NivaraFrame Normalize(this NivaraFrame frame, params string[]? columns)
    {
        if (frame == null) throw new ArgumentNullException(nameof(frame));

        // If no columns specified, normalize all supported numeric columns
        if (columns is null || columns.Length == 0)
            columns = frame.ColumnNames.Where(name => IsNumericColumn(frame, name)).ToArray();

        // Create a set of columns to normalize for quick lookup
        var columnsToNormalize = new HashSet<string>(columns, StringComparer.OrdinalIgnoreCase);

        // Build a new list of columns, normalizing as needed
        var newColumns = new List<(string Name, IColumn Column)>();

        foreach (var columnName in frame.ColumnNames)
        {
            IColumn resultColumn;

            if (columnsToNormalize.Contains(columnName))
            {
                if (!IsNumericColumn(frame, columnName))
                    throw new NotSupportedException($"Normalization for column '{columnName}' of type {frame.Schema.GetColumnType(columnName)} is not supported. Only INumber<T> columns can be normalized.");

                resultColumn = NormalizeColumn(frame, columnName);
            }
            else
            {
                resultColumn = frame.GetColumn(columnName);
            }

            newColumns.Add((columnName, resultColumn));
        }

        return new NivaraFrame(newColumns);
    }

    /// <summary>
    /// Determines whether a column is a supported numeric type for normalization
    /// (implements <see cref="INumber{T}"/>, excluding a small blocklist).
    /// </summary>
    private static bool IsNumericColumn(NivaraFrame frame, string columnName)
        => IsSupportedNumericType(frame.Schema.GetColumnType(columnName));

    private static bool IsSupportedNumericType(Type columnType)
        => !columnType.IsEnum &&
           !ExcludedNumericTypes.Contains(columnType) &&
           columnType.GetInterfaces().Any(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(INumber<>));

    /// <summary>
    /// <see cref="INumber{T}"/> implementers that should not be normalized: <c>char</c> is a
    /// surprising candidate, and <c>BigInteger</c>/<c>Int128</c>/<c>UInt128</c> are not common
    /// frame columns. Kept explicit so the interface predicate stays predictable.
    /// </summary>
    private static readonly HashSet<Type> ExcludedNumericTypes = [typeof(char), typeof(BigInteger), typeof(Int128), typeof(UInt128)];

    /// <summary>
    /// Normalizes a single column to zero mean and unit variance, skipping null values.
    /// Dispatches through a per-type cached compiled delegate: the interface predicate runs
    /// once per column type, after which every call is a direct delegate invocation.
    /// </summary>
    private static IColumn NormalizeColumn(NivaraFrame frame, string columnName)
    {
        var columnType = frame.Schema.GetColumnType(columnName);
        if (!_normalizeDispatch.TryGetValue(columnType, out var normalizer))
        {
            normalizer = BuildNormalizer(columnType);
            _normalizeDispatch[columnType] = normalizer;
        }
        return normalizer(frame, columnName);
    }

    private static readonly ConcurrentDictionary<Type, Func<NivaraFrame, string, IColumn>> _normalizeDispatch = new();

    private static Func<NivaraFrame, string, IColumn> BuildNormalizer(Type columnType)
    {
        if (!IsSupportedNumericType(columnType))
            throw new NotSupportedException($"Normalization for type {columnType} is not supported. Only INumber<T> columns can be normalized.");

        var isFloatType = columnType.GetInterfaces().Any(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(IFloatingPointIeee754<>));
        var coreName = isFloatType ? nameof(NormalizeFloatCore) : nameof(NormalizeIntegerCore);

        var coreMethod = typeof(NivaraFrameExtensions)
            .GetMethod(coreName, BindingFlags.Static | BindingFlags.NonPublic)!
            .MakeGenericMethod(columnType);

        var frame = Expression.Parameter(typeof(NivaraFrame), "frame");
        var name = Expression.Parameter(typeof(string), "name");
        var typedGetColumn = typeof(NivaraFrame)
            .GetMethods(BindingFlags.Instance | BindingFlags.Public)
            .First(m => m.Name == nameof(NivaraFrame.GetColumn) && m.IsGenericMethodDefinition)
            .MakeGenericMethod(columnType);

        var call = Expression.Call(null, coreMethod, Expression.Call(frame, typedGetColumn, name));
        return Expression.Lambda<Func<NivaraFrame, string, IColumn>>(call, frame, name).Compile();
    }

    /// <summary>
    /// SIMD z-score core for IEEE-754 float types. Uses <see cref="TensorPrimitives"/> for
    /// statistics and transform; null values are excluded from the statistics and preserved in
    /// the result via the null mask.
    /// </summary>
    private static IColumn NormalizeFloatCore<T>(NivaraColumn<T> column)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (column.TryGetSpan(out var span))
        {
            var normalized = new T[span.Length];
            span.CopyTo(normalized);
            return TensorsHelper.TryNormalizeInPlace(normalized) ? NivaraColumn<T>.Create(normalized) : column;
        }

        var length = column.Length;
        var values = new T[length];
        var nullMask = new bool[length];
        var packed = new T[length];
        int count = 0;

        for (int i = 0; i < length; i++)
        {
            if (column.IsNull(i)) { nullMask[i] = true; continue; }

            var value = column[i];
            values[i] = value;
            packed[count++] = value;
        }

        if (count > 0 && TensorsHelper.TryNormalizeInPlace(packed.AsSpan(0, count)))
        {
            int d = 0;
            for (int i = 0; i < length; i++)
                if (!nullMask[i]) values[i] = packed[d++];
        }

        return NivaraColumn<T>.CreateFromSpans(values, nullMask);
    }

    /// <summary>
    /// Generic z-score core for any <see cref="INumber{T}"/> column. The output promotes to
    /// <c>NivaraColumn&lt;double&gt;</c> because z-scores are fractional; conversion and
    /// statistics run in <c>double</c> via <see cref="TensorsHelper.TryNormalizeToDouble{T}"/>.
    /// </summary>
    private static IColumn NormalizeIntegerCore<T>(NivaraColumn<T> column)
        where T : struct, INumber<T>
    {
        if (column.TryGetSpan(out var span))
        {
            var normalized = new double[span.Length];
            return TensorsHelper.TryNormalizeToDouble(span, normalized) ? NivaraColumn<double>.Create(normalized) : column;
        }

        var length = column.Length;
        var values = new double[length];
        var nullMask = new bool[length];
        var packed = new T[length];
        int count = 0;

        for (int i = 0; i < length; i++)
        {
            if (column.IsNull(i)) { nullMask[i] = true; continue; }

            var value = column[i];
            values[i] = double.CreateChecked(value);
            packed[count++] = value;
        }

        if (count > 0)
        {
            var destination = new double[count];
            if (TensorsHelper.TryNormalizeToDouble(packed.AsSpan(0, count), destination))
            {
                int d = 0;
                for (int i = 0; i < length; i++)
                    if (!nullMask[i]) values[i] = destination[d++];
            }
        }

        return NivaraColumn<double>.CreateFromSpans(values, nullMask);
    }

    #endregion
}
