using Microsoft.Extensions.AI;

namespace Nivara.AI;

/// <summary>
/// Supplies the vector emitted for a null embedding row when crossing into
/// <see cref="Microsoft.Extensions.AI"/>, which has no null vector.
/// </summary>
/// <param name="rowIndex">Row of the source frame whose embedding is null.</param>
/// <param name="dimension">Embedding dimension, inferred from the non-null rows or supplied explicitly.</param>
/// <returns>The vector to emit for that row; its length must equal <paramref name="dimension"/>.</returns>
public delegate float[] NullEmbeddingReplacement(int rowIndex, int dimension);

/// <summary>
/// Repairs a row whose dimension cells are only partly null, a shape no <c>float[]</c> can represent.
/// </summary>
/// <param name="rowIndex">Row holding the partially null vector.</param>
/// <param name="values">The row's stored values. Entries listed in <paramref name="nullDimensions"/> are placeholders.</param>
/// <param name="nullDimensions">Dimension indices that are null in that row.</param>
/// <returns>A complete vector, or <see langword="null"/> to collapse the whole row to a null embedding.</returns>
public delegate float[]? PartialEmbeddingRepair(int rowIndex, float[] values, int[] nullDimensions);

/// <summary>
/// The documented embedding convention: a <c>NivaraColumn&lt;float[]&gt;</c> for storage and
/// Microsoft.Extensions.AI interop, and a flat <c>[rows, dim]</c> frame of <c>E0..E{n-1}</c> float
/// columns for scoring with the existing <see cref="NivaraFrame.RowCosineSimilarity{T}"/> and
/// <see cref="NivaraFrame.RowDot{T}"/> kernels. Nivara owns the table; the BCL owns the math.
/// </summary>
public static class NivaraEmbeddingExtensions
{
    /// <summary>
    /// Lands a generated embedding set as the documented convention: one <c>float[]</c> column.
    /// </summary>
    /// <param name="generated">The embeddings returned by an <see cref="IEmbeddingGenerator{TInput,TEmbedding}"/>.</param>
    /// <param name="embeddingColumnName">Name of the column holding the vectors.</param>
    /// <param name="idColumnName">Optional name of a column carrying row identity alongside the vectors.</param>
    /// <param name="ids">Identity values, one per embedding. Required when <paramref name="idColumnName"/> is set.</param>
    /// <returns>A frame with the vector column and, when requested, the identity column.</returns>
    public static NivaraFrame ToNivaraFrame(
        this GeneratedEmbeddings<Embedding<float>> generated,
        string embeddingColumnName = "Embedding",
        string? idColumnName = null,
        IEnumerable<object>? ids = null)
    {
        ArgumentNullException.ThrowIfNull(generated);
        ArgumentException.ThrowIfNullOrWhiteSpace(embeddingColumnName);

        var vectors = new float[generated.Count][];
        for (var i = 0; i < generated.Count; i++)
            vectors[i] = generated[i].Vector.ToArray();

        var embeddingColumn = NivaraColumn<float[]>.Create(vectors);

        if (idColumnName is null)
        {
            if (ids is not null)
                throw new ArgumentException(
                    $"'{nameof(ids)}' requires '{nameof(idColumnName)}' to name the identity column.",
                    nameof(ids));

            return NivaraFrame.Create((embeddingColumnName, (IColumn)embeddingColumn));
        }

        ArgumentException.ThrowIfNullOrWhiteSpace(idColumnName);
        ArgumentNullException.ThrowIfNull(ids, nameof(ids));

        var idValues = ids as object[] ?? [.. ids];
        if (idValues.Length != generated.Count)
            throw new ArgumentException(
                $"Identity count ({idValues.Length}) must match embedding count ({generated.Count}).",
                nameof(ids));

        var idColumn = NivaraColumn<object>.Create(idValues);
        return NivaraFrame.Create(
            (embeddingColumnName, (IColumn)embeddingColumn),
            (idColumnName, (IColumn)idColumn));
    }

    /// <summary>
    /// Converts the convention's vector column back into
    /// <see cref="GeneratedEmbeddings{TEmbedding}"/> for Microsoft.Extensions.AI consumers.
    /// </summary>
    /// <param name="frame">Frame holding the <c>float[]</c> column.</param>
    /// <param name="embeddingColumnName">Name of the column holding the vectors.</param>
    /// <param name="nullReplacement">
    /// Supplies a vector for each null row. Required when the column contains nulls, because
    /// <see cref="Microsoft.Extensions.AI"/> cannot represent a null embedding; without it a null row throws.
    /// </param>
    /// <param name="dimension">Embedding dimension. Inferred from the first non-null vector when omitted.</param>
    /// <returns>The vectors, in row order.</returns>
    public static GeneratedEmbeddings<Embedding<float>> ToGeneratedEmbeddings(
        this NivaraFrame frame,
        string embeddingColumnName = "Embedding",
        NullEmbeddingReplacement? nullReplacement = null,
        int? dimension = null)
    {
        ArgumentNullException.ThrowIfNull(frame);

        var column = frame.GetColumn<float[]>(embeddingColumnName);
        if (column.Length == 0)
            return new GeneratedEmbeddings<Embedding<float>>();

        var resolvedDimension = resolveDimension(column, dimension, embeddingColumnName);
        var result = new GeneratedEmbeddings<Embedding<float>>(column.Length);

        for (var row = 0; row < column.Length; row++)
        {
            if (column.IsNull(row))
            {
                if (nullReplacement is null)
                    throw new InvalidOperationException(
                        $"Row {row} of column '{embeddingColumnName}' holds a null embedding, and "
                        + $"Microsoft.Extensions.AI has no null vector. Supply a '{nameof(nullReplacement)}' "
                        + "policy, or filter the row out of the frame before converting.");

                var replacement = nullReplacement(row, resolvedDimension);
                if (replacement is null || replacement.Length != resolvedDimension)
                    throw new InvalidOperationException(
                        $"The '{nameof(nullReplacement)}' supplied for row {row} returned a vector of length "
                        + $"{replacement?.Length.ToString() ?? "null"}, expected {resolvedDimension}.");

                result.Add(new Embedding<float>(replacement));
                continue;
            }

            var vector = column[row];
            if (vector.Length != resolvedDimension)
                throw new InvalidOperationException(
                    $"Row {row} of column '{embeddingColumnName}' has {vector.Length} dimensions, "
                    + $"but the embedding dimension is {resolvedDimension}. Embeddings in one column must share a dimension.");

            result.Add(new Embedding<float>(vector));
        }

        return result;
    }

    /// <summary>
    /// Converts the convention's vector column into the flat <c>[rows, dim]</c> form the row-scoring
    /// kernels understand: one float column per dimension, named <c>E0..E{n-1}</c>.
    /// </summary>
    /// <param name="frame">Frame holding the <c>float[]</c> column.</param>
    /// <param name="embeddingColumnName">Name of the column holding the vectors.</param>
    /// <param name="columnPrefix">Prefix of the generated dimension columns.</param>
    /// <param name="dimension">Embedding dimension. Inferred from the first non-null vector when omitted.</param>
    /// <returns>
    /// A frame containing only the dimension columns. Metadata and labels stay on the source frame and
    /// are passed to the scoring methods as <c>labels</c>.
    /// </returns>
    public static NivaraFrame ToEmbeddingMatrix(
        this NivaraFrame frame,
        string embeddingColumnName = "Embedding",
        string columnPrefix = "E",
        int? dimension = null)
    {
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentException.ThrowIfNullOrWhiteSpace(columnPrefix);

        var column = frame.GetColumn<float[]>(embeddingColumnName);
        var resolvedDimension = resolveDimension(column, dimension, embeddingColumnName);
        var rowCount = column.Length;

        var data = new float[resolvedDimension][];
        var mask = new bool[resolvedDimension][];
        for (var d = 0; d < resolvedDimension; d++)
        {
            data[d] = new float[rowCount];
            mask[d] = new bool[rowCount];
        }

        for (var row = 0; row < rowCount; row++)
        {
            if (column.IsNull(row))
            {
                for (var d = 0; d < resolvedDimension; d++)
                    mask[d][row] = true;

                continue;
            }

            var vector = column[row];
            if (vector.Length != resolvedDimension)
                throw new InvalidOperationException(
                    $"Row {row} of column '{embeddingColumnName}' has {vector.Length} dimensions, "
                    + $"but the embedding dimension is {resolvedDimension}. Embeddings in one column must share a dimension.");

            for (var d = 0; d < resolvedDimension; d++)
                data[d][row] = vector[d];
        }

        var namedColumns = new (string Name, IColumn Column)[resolvedDimension];
        for (var d = 0; d < resolvedDimension; d++)
            namedColumns[d] = ($"{columnPrefix}{d}", NivaraColumn<float>.CreateFromSpans(data[d], mask[d]));

        return NivaraFrame.Create(namedColumns);
    }

    /// <summary>
    /// Converts a flat <c>[rows, dim]</c> frame back into the convention's single <c>float[]</c> column.
    /// </summary>
    /// <param name="matrixFrame">Frame holding the <c>E0..E{n-1}</c> dimension columns.</param>
    /// <param name="embeddingColumnName">Name of the column the vectors are written to.</param>
    /// <param name="columnPrefix">Prefix of the dimension columns.</param>
    /// <param name="dimension">Number of dimension columns. Inferred from the column names when omitted.</param>
    /// <param name="repair">
    /// Resolves a row whose dimension cells are only partly null. Required when such a row exists; a
    /// fully null row becomes a null embedding without consulting it.
    /// </param>
    /// <returns>A frame with a single <c>float[]</c> column.</returns>
    public static NivaraFrame ToEmbeddingFrame(
        this NivaraFrame matrixFrame,
        string embeddingColumnName = "Embedding",
        string columnPrefix = "E",
        int? dimension = null,
        PartialEmbeddingRepair? repair = null)
    {
        ArgumentNullException.ThrowIfNull(matrixFrame);
        ArgumentException.ThrowIfNullOrWhiteSpace(columnPrefix);
        ArgumentException.ThrowIfNullOrWhiteSpace(embeddingColumnName);

        var resolvedDimension = dimension ?? inferMatrixDimension(matrixFrame, columnPrefix);
        if (resolvedDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(dimension), resolvedDimension, "Embedding dimension must be positive.");

        var dimensionColumns = new NivaraColumn<float>[resolvedDimension];
        for (var d = 0; d < resolvedDimension; d++)
        {
            var name = $"{columnPrefix}{d}";
            if (!matrixFrame.HasColumn(name))
                throw new ArgumentException(
                    $"Frame has no column '{name}'. Expected {resolvedDimension} dimension columns named "
                    + $"'{columnPrefix}0'..'{columnPrefix}{resolvedDimension - 1}'.",
                    nameof(matrixFrame));

            dimensionColumns[d] = matrixFrame.GetColumn<float>(name);
        }

        var rowCount = dimensionColumns[0].Length;
        var vectors = new float[rowCount][];

        for (var row = 0; row < rowCount; row++)
        {
            var nullCount = 0;
            for (var d = 0; d < resolvedDimension; d++)
            {
                if (dimensionColumns[d].IsNull(row))
                    nullCount++;
            }

            if (nullCount == 0)
            {
                var vector = new float[resolvedDimension];
                for (var d = 0; d < resolvedDimension; d++)
                    vector[d] = dimensionColumns[d][row];

                vectors[row] = vector;
                continue;
            }

            if (nullCount == resolvedDimension)
            {
                vectors[row] = null!;
                continue;
            }

            if (repair is null)
                throw new InvalidOperationException(
                    $"Row {row} has {nullCount} of {resolvedDimension} dimension cells null, a shape no float[] can "
                    + $"represent. Supply a '{nameof(repair)}' policy, or make the row fully null.");

            var values = new float[resolvedDimension];
            var nullDimensions = new int[nullCount];
            var next = 0;
            for (var d = 0; d < resolvedDimension; d++)
            {
                if (dimensionColumns[d].IsNull(row))
                    nullDimensions[next++] = d;
                else
                    values[d] = dimensionColumns[d][row];
            }

            var repaired = repair(row, values, nullDimensions);
            if (repaired is null)
            {
                vectors[row] = null!;
                continue;
            }

            if (repaired.Length != resolvedDimension)
                throw new InvalidOperationException(
                    $"The '{nameof(repair)}' supplied for row {row} returned a vector of length "
                    + $"{repaired.Length}, expected {resolvedDimension}.");

            vectors[row] = repaired;
        }

        var column = NivaraColumn<float[]>.Create(vectors);
        return NivaraFrame.Create((embeddingColumnName, (IColumn)column));
    }

    /// <summary>
    /// Scores each row of the convention's vector column against a labeled query using cosine
    /// similarity, delegating to <see cref="NivaraFrame.RowCosineSimilarity{T}"/>.
    /// A null embedding scores null; a null query element scores every row null.
    /// </summary>
    /// <param name="frame">Frame holding the <c>float[]</c> column.</param>
    /// <param name="embeddingColumnName">Name of the column holding the vectors.</param>
    /// <param name="query">The query vector, one element per dimension. A null element masks all scores.</param>
    /// <param name="labels">Optional row labels, carried onto the returned series.</param>
    /// <returns>One score per row.</returns>
    public static NivaraSeries<float> RowCosineSimilarity(
        this NivaraFrame frame,
        string embeddingColumnName,
        NivaraSeries<float> query,
        IColumn? labels = null)
        => score(frame, embeddingColumnName, query, labels, cosine: true);

    /// <summary>
    /// Scores each row of the convention's vector column against a query using cosine similarity,
    /// delegating to <see cref="NivaraFrame.RowCosineSimilarity{T}"/>.
    /// </summary>
    /// <param name="frame">Frame holding the <c>float[]</c> column.</param>
    /// <param name="embeddingColumnName">Name of the column holding the vectors.</param>
    /// <param name="query">The query vector. <see langword="null"/> masks every score.</param>
    /// <param name="labels">Optional row labels, carried onto the returned series.</param>
    /// <returns>One score per row.</returns>
    public static NivaraSeries<float> RowCosineSimilarity(
        this NivaraFrame frame,
        string embeddingColumnName,
        float[]? query,
        IColumn? labels = null)
        => score(frame, embeddingColumnName, toQuerySeries(frame, embeddingColumnName, query), labels, cosine: true);

    /// <summary>
    /// Scores each row of the convention's vector column against a labeled query using the dot
    /// product, delegating to <see cref="NivaraFrame.RowDot{T}"/>.
    /// A null embedding scores null; a null query element scores every row null.
    /// </summary>
    /// <param name="frame">Frame holding the <c>float[]</c> column.</param>
    /// <param name="embeddingColumnName">Name of the column holding the vectors.</param>
    /// <param name="query">The query vector, one element per dimension. A null element masks all scores.</param>
    /// <param name="labels">Optional row labels, carried onto the returned series.</param>
    /// <returns>One score per row.</returns>
    public static NivaraSeries<float> RowDot(
        this NivaraFrame frame,
        string embeddingColumnName,
        NivaraSeries<float> query,
        IColumn? labels = null)
        => score(frame, embeddingColumnName, query, labels, cosine: false);

    /// <summary>
    /// Scores each row of the convention's vector column against a query using the dot product,
    /// delegating to <see cref="NivaraFrame.RowDot{T}"/>.
    /// </summary>
    /// <param name="frame">Frame holding the <c>float[]</c> column.</param>
    /// <param name="embeddingColumnName">Name of the column holding the vectors.</param>
    /// <param name="query">The query vector. <see langword="null"/> masks every score.</param>
    /// <param name="labels">Optional row labels, carried onto the returned series.</param>
    /// <returns>One score per row.</returns>
    public static NivaraSeries<float> RowDot(
        this NivaraFrame frame,
        string embeddingColumnName,
        float[]? query,
        IColumn? labels = null)
        => score(frame, embeddingColumnName, toQuerySeries(frame, embeddingColumnName, query), labels, cosine: false);

    /// <summary>
    /// Generates embeddings for <paramref name="values"/> and lands them as the documented convention.
    /// </summary>
    /// <typeparam name="TInput">The input type accepted by the generator.</typeparam>
    /// <param name="generator">The embedding generator.</param>
    /// <param name="values">The inputs to embed.</param>
    /// <param name="embeddingColumnName">Name of the column holding the vectors.</param>
    /// <param name="cancellationToken">Cancels the generation.</param>
    /// <returns>A frame with one <c>float[]</c> column, one row per input.</returns>
    public static async Task<NivaraFrame> GenerateFrameAsync<TInput>(
        this IEmbeddingGenerator<TInput, Embedding<float>> generator,
        IEnumerable<TInput> values,
        string embeddingColumnName = "Embedding",
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(generator);
        ArgumentNullException.ThrowIfNull(values);

        var generated = await generator
            .GenerateAsync(values, cancellationToken: cancellationToken)
            .ConfigureAwait(false);

        return generated.ToNivaraFrame(embeddingColumnName);
    }

    static NivaraSeries<float> score(
        NivaraFrame frame,
        string embeddingColumnName,
        NivaraSeries<float> query,
        IColumn? labels,
        bool cosine)
    {
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentNullException.ThrowIfNull(query);

        var column = frame.GetColumn<float[]>(embeddingColumnName);
        var columnDimension = tryResolveDimension(column, embeddingColumnName);
        if (columnDimension is { } known && query.Length != known)
            throw new ArgumentException(
                $"Query length ({query.Length}) must match the embedding dimension ({known}).",
                nameof(query));

        var resolvedDimension = columnDimension ?? query.Length;
        if (resolvedDimension <= 0)
            throw new ArgumentException(
                $"Query length must be positive when column '{embeddingColumnName}' holds no non-null embedding.",
                nameof(query));

        var matrix = frame.ToEmbeddingMatrix(embeddingColumnName, dimension: resolvedDimension);
        return cosine
            ? matrix.RowCosineSimilarity(query, labels)
            : matrix.RowDot(query, labels);
    }

    static NivaraSeries<float> toQuerySeries(NivaraFrame frame, string embeddingColumnName, float[]? query)
    {
        ArgumentNullException.ThrowIfNull(frame);

        if (query is not null)
            return NivaraSeries<float>.Create(query);

        // An all-null query masks every score, so its length never reaches the kernel. One
        // dimension is enough to build the matrix when the column cannot supply a dimension.
        var dimension = tryResolveDimension(
            frame.GetColumn<float[]>(embeddingColumnName),
            embeddingColumnName) ?? 1;

        var values = new float[dimension];
        var mask = new bool[dimension];
        Array.Fill(mask, true);
        return new NivaraSeries<float>(NivaraColumn<float>.CreateFromSpans(values, mask));
    }

    static int resolveDimension(NivaraColumn<float[]> column, int? dimension, string embeddingColumnName)
    {
        if (dimension is { } explicitDimension)
        {
            if (explicitDimension <= 0)
                throw new ArgumentOutOfRangeException(nameof(dimension), explicitDimension, "Embedding dimension must be positive.");

            return explicitDimension;
        }

        return tryResolveDimension(column, embeddingColumnName)
            ?? throw new InvalidOperationException(
                $"Column '{embeddingColumnName}' holds no non-null embedding, so the dimension cannot be inferred. "
                + "Pass 'dimension' explicitly.");
    }

    static int? tryResolveDimension(NivaraColumn<float[]> column, string embeddingColumnName)
    {
        for (var row = 0; row < column.Length; row++)
        {
            if (column.IsNull(row))
                continue;

            var vector = column[row];
            if (vector.Length == 0)
                throw new InvalidOperationException(
                    $"Row {row} of column '{embeddingColumnName}' holds an empty embedding, so the dimension cannot be inferred. "
                    + "Pass 'dimension' explicitly.");

            return vector.Length;
        }

        return null;
    }

    static int inferMatrixDimension(NivaraFrame matrixFrame, string columnPrefix)
    {
        var dimension = 0;
        while (matrixFrame.HasColumn($"{columnPrefix}{dimension}"))
            dimension++;

        if (dimension == 0)
            throw new ArgumentException(
                $"Frame has no column named '{columnPrefix}0', so the embedding dimension cannot be inferred. "
                + "Pass 'dimension' explicitly.",
                nameof(matrixFrame));

        return dimension;
    }
}
