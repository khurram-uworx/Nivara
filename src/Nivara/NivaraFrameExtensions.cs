using Nivara.Exceptions;
using Nivara.Expressions;

namespace Nivara;

/// <summary>
/// Extension methods for NivaraFrame to support transformations and projections
/// </summary>
public static class NivaraFrameExtensions
{
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
}