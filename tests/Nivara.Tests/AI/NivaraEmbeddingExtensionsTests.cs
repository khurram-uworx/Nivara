using Microsoft.Extensions.AI;
using Nivara.AI;
using NUnit.Framework;
using System.Numerics.Tensors;

namespace Nivara.Tests.AI;

/// <summary>
/// Tests for the documented embedding convention: a <c>float[]</c> column for storage and
/// Microsoft.Extensions.AI interop, and a flat <c>[rows, dim]</c> frame for the row-scoring kernels.
/// </summary>
[TestFixture]
public class NivaraEmbeddingExtensionsTests
{
    const string Embedding = "Embedding";

    static GeneratedEmbeddings<Embedding<float>> Generated(params float[][] vectors)
    {
        var generated = new GeneratedEmbeddings<Embedding<float>>(vectors.Length);
        foreach (var vector in vectors)
            generated.Add(new Embedding<float>(vector));

        return generated;
    }

    static NivaraFrame VectorFrame(params float[]?[] vectors)
    {
        var values = new float[vectors.Length][];
        for (var i = 0; i < vectors.Length; i++)
            values[i] = vectors[i]!;

        return NivaraFrame.Create((Embedding, (IColumn)NivaraColumn<float[]>.CreateForReferenceType(values)));
    }

    static NivaraColumn<float[]> Vectors(NivaraFrame frame) => frame.GetColumn<float[]>(Embedding);

    // ── MEAI → frame ────────────────────────────────────────────────

    [Test]
    public void ToNivaraFrame_CarriesVectorsInRowOrder()
    {
        var generated = Generated([1f, 2f], [3f, 4f]);

        var frame = generated.ToNivaraFrame(Embedding);

        Assert.That(frame.RowCount, Is.EqualTo(2));
        Assert.That(frame.ColumnCount, Is.EqualTo(1));
        Assert.That(Vectors(frame).HasNulls, Is.False);
        Assert.That(Vectors(frame)[0], Is.EqualTo(new[] { 1f, 2f }));
        Assert.That(Vectors(frame)[1], Is.EqualTo(new[] { 3f, 4f }));
    }

    [Test]
    public void ToNivaraFrame_WithIds_AddsIdentityColumn()
    {
        var generated = Generated([1f, 2f], [3f, 4f]);

        var frame = generated.ToNivaraFrame(Embedding, "ChunkId", ["c0", "c1"]);

        Assert.That(frame.ColumnCount, Is.EqualTo(2));
        Assert.That(frame.GetColumn<object>("ChunkId")[0], Is.EqualTo("c0"));
        Assert.That(frame.GetColumn<object>("ChunkId")[1], Is.EqualTo("c1"));
    }

    [Test]
    public void ToNivaraFrame_IdCountMismatch_Throws()
    {
        var generated = Generated([1f, 2f], [3f, 4f]);

        var ex = Assert.Throws<ArgumentException>(() =>
            generated.ToNivaraFrame(Embedding, "ChunkId", ["c0"]));

        Assert.That(ex!.Message, Does.Contain("must match embedding count"));
    }

    [Test]
    public void ToNivaraFrame_IdsWithoutColumnName_Throws()
    {
        var generated = Generated([1f, 2f]);

        var ex = Assert.Throws<ArgumentException>(() =>
            generated.ToNivaraFrame(Embedding, ids: ["c0"]));

        Assert.That(ex!.Message, Does.Contain("idColumnName"));
    }

    [Test]
    public void ToNivaraFrame_EmptyInput_ProducesEmptyFrame()
    {
        var frame = Generated().ToNivaraFrame(Embedding);

        Assert.That(frame.RowCount, Is.EqualTo(0));
        Assert.That(Vectors(frame).HasNulls, Is.False);
    }

    [Test]
    public void ToGeneratedEmbeddings_EmptyColumn_ReturnsEmptyResult()
    {
        var frame = VectorFrame();

        var generated = frame.ToGeneratedEmbeddings(Embedding);

        Assert.That(generated.Count, Is.EqualTo(0));
    }

    // ── float[] column → flat [rows, dim] frame ─────────────────────

    [Test]
    public void ToEmbeddingMatrix_ProducesPrefixedDimensionColumns()
    {
        var frame = VectorFrame([1f, 2f, 3f], [4f, 5f, 6f]);

        var matrix = frame.ToEmbeddingMatrix(Embedding);

        Assert.That(matrix.ColumnCount, Is.EqualTo(3));
        Assert.That(matrix.ColumnNames, Is.EqualTo(new[] { "E0", "E1", "E2" }));
        Assert.That(matrix.GetColumn<float>("E0")[0], Is.EqualTo(1f));
        Assert.That(matrix.GetColumn<float>("E2")[1], Is.EqualTo(6f));
        Assert.That(matrix.GetColumn<float>("E1").HasNulls, Is.False);
    }

    [Test]
    public void ToEmbeddingMatrix_HonoursColumnPrefix()
    {
        var frame = VectorFrame([1f, 2f]);

        var matrix = frame.ToEmbeddingMatrix(Embedding, columnPrefix: "Dim");

        Assert.That(matrix.ColumnNames, Is.EqualTo(new[] { "Dim0", "Dim1" }));
    }

    [Test]
    public void ToEmbeddingMatrix_NullRow_ProducesAllNullRow()
    {
        var frame = VectorFrame([1f, 2f], null, [5f, 6f]);

        var matrix = frame.ToEmbeddingMatrix(Embedding);

        Assert.That(matrix.RowCount, Is.EqualTo(3));
        Assert.That(matrix.GetColumn<float>("E0").IsNull(1), Is.True);
        Assert.That(matrix.GetColumn<float>("E1").IsNull(1), Is.True);
        Assert.That(matrix.GetColumn<float>("E0").IsNull(0), Is.False);
        Assert.That(matrix.GetColumn<float>("E0")[2], Is.EqualTo(5f));
    }

    [Test]
    public void ToEmbeddingMatrix_HeterogeneousLengths_ThrowsWithRowIndex()
    {
        var frame = VectorFrame([1f, 2f], [1f, 2f, 3f]);

        var ex = Assert.Throws<InvalidOperationException>(() => frame.ToEmbeddingMatrix(Embedding));

        Assert.That(ex!.Message, Does.Contain("Row 1"));
        Assert.That(ex.Message, Does.Contain("3 dimensions"));
    }

    [Test]
    public void ToEmbeddingMatrix_AllNullWithoutDimension_Throws()
    {
        var frame = VectorFrame(null, null);

        var ex = Assert.Throws<InvalidOperationException>(() => frame.ToEmbeddingMatrix(Embedding));

        Assert.That(ex!.Message, Does.Contain("dimension"));
    }

    [Test]
    public void ToEmbeddingMatrix_AllNullWithDimension_Succeeds()
    {
        var frame = VectorFrame(null, null);

        var matrix = frame.ToEmbeddingMatrix(Embedding, dimension: 2);

        Assert.That(matrix.ColumnCount, Is.EqualTo(2));
        Assert.That(matrix.GetColumn<float>("E0").IsNull(0), Is.True);
        Assert.That(matrix.GetColumn<float>("E1").IsNull(1), Is.True);
    }

    // ── flat frame → float[] column ─────────────────────────────────

    [Test]
    public void ToEmbeddingFrame_RoundTripsValuesExactly()
    {
        var original = VectorFrame([1.5f, 2.25f], [-3f, 4f]);

        var rebuilt = original.ToEmbeddingMatrix(Embedding).ToEmbeddingFrame(Embedding);

        Assert.That(Vectors(rebuilt).HasNulls, Is.False);
        Assert.That(Vectors(rebuilt)[0], Is.EqualTo(new[] { 1.5f, 2.25f }));
        Assert.That(Vectors(rebuilt)[1], Is.EqualTo(new[] { -3f, 4f }));
    }

    [Test]
    public void ToEmbeddingFrame_NullRow_RoundTripsAsNullEmbedding()
    {
        var original = VectorFrame([1f, 2f], null);

        var rebuilt = original.ToEmbeddingMatrix(Embedding).ToEmbeddingFrame(Embedding);

        Assert.That(Vectors(rebuilt).IsNull(1), Is.True);
        Assert.That(Vectors(rebuilt)[0], Is.EqualTo(new[] { 1f, 2f }));
    }

    [Test]
    public void ToEmbeddingFrame_FullyNullRow_NeedsNoRepair()
    {
        var matrix = NivaraFrame.Create(
            ("E0", (IColumn)NivaraColumnFactory.CreateFromNullable(new float?[] { null })),
            ("E1", (IColumn)NivaraColumnFactory.CreateFromNullable(new float?[] { null })));

        var rebuilt = matrix.ToEmbeddingFrame(Embedding);

        Assert.That(Vectors(rebuilt).IsNull(0), Is.True);
    }

    [Test]
    public void ToEmbeddingFrame_PartialNullWithoutRepair_Throws()
    {
        var matrix = NivaraFrame.Create(
            ("E0", (IColumn)NivaraColumn<float>.Create(new[] { 1f })),
            ("E1", (IColumn)NivaraColumnFactory.CreateFromNullable(new float?[] { null })));

        var ex = Assert.Throws<InvalidOperationException>(() => matrix.ToEmbeddingFrame(Embedding));

        Assert.That(ex!.Message, Does.Contain("Row 0"));
        Assert.That(ex.Message, Does.Contain("repair"));
    }

    [Test]
    public void ToEmbeddingFrame_PartialNullWithRepair_FillsNullDimensions()
    {
        var matrix = NivaraFrame.Create(
            ("E0", (IColumn)NivaraColumn<float>.Create(new[] { 1f })),
            ("E1", (IColumn)NivaraColumnFactory.CreateFromNullable(new float?[] { null })));

        float[]? seenValues = null;
        int[]? seenNulls = null;
        var rebuilt = matrix.ToEmbeddingFrame(Embedding, repair: (row, values, nulls) =>
        {
            Assert.That(row, Is.EqualTo(0));
            seenValues = values;
            seenNulls = nulls;

            var filled = (float[])values.Clone();
            foreach (var dimension in nulls)
                filled[dimension] = 42f;

            return filled;
        });

        Assert.That(seenValues![0], Is.EqualTo(1f));
        Assert.That(seenNulls, Is.EqualTo(new[] { 1 }));
        Assert.That(Vectors(rebuilt)[0], Is.EqualTo(new[] { 1f, 42f }));
    }

    [Test]
    public void ToEmbeddingFrame_RepairReturnsNull_CollapsesRowToNullEmbedding()
    {
        var matrix = NivaraFrame.Create(
            ("E0", (IColumn)NivaraColumn<float>.Create(new[] { 1f })),
            ("E1", (IColumn)NivaraColumnFactory.CreateFromNullable(new float?[] { null })));

        var rebuilt = matrix.ToEmbeddingFrame(Embedding, repair: (_, _, _) => null);

        Assert.That(Vectors(rebuilt).IsNull(0), Is.True);
    }

    [Test]
    public void ToEmbeddingFrame_RepairReturnsWrongLength_Throws()
    {
        var matrix = NivaraFrame.Create(
            ("E0", (IColumn)NivaraColumn<float>.Create(new[] { 1f })),
            ("E1", (IColumn)NivaraColumnFactory.CreateFromNullable(new float?[] { null })));

        var ex = Assert.Throws<InvalidOperationException>(() =>
            matrix.ToEmbeddingFrame(Embedding, repair: (_, _, _) => new[] { 1f, 2f, 3f }));

        Assert.That(ex!.Message, Does.Contain("length 3"));
    }

    [Test]
    public void ToEmbeddingFrame_MissingDimensionColumn_Throws()
    {
        var matrix = NivaraFrame.Create(
            ("E0", (IColumn)NivaraColumn<float>.Create(new[] { 1f })));

        var ex = Assert.Throws<ArgumentException>(() =>
            matrix.ToEmbeddingFrame(Embedding, dimension: 2));

        Assert.That(ex!.Message, Does.Contain("E1"));
    }

    [Test]
    public void ToEmbeddingFrame_NoDimensionColumns_Throws()
    {
        var matrix = NivaraFrame.Create(
            ("A", (IColumn)NivaraColumn<float>.Create(new[] { 1f })));

        var ex = Assert.Throws<ArgumentException>(() => matrix.ToEmbeddingFrame(Embedding));

        Assert.That(ex!.Message, Does.Contain("E0"));
    }

    // ── frame → MEAI ────────────────────────────────────────────────

    [Test]
    public void ToGeneratedEmbeddings_RoundTripsValuesExactly()
    {
        var frame = VectorFrame([1.5f, 2.25f], [-3f, 4f]);

        var generated = frame.ToGeneratedEmbeddings(Embedding);

        Assert.That(generated.Count, Is.EqualTo(2));
        Assert.That(generated[0].Vector.ToArray(), Is.EqualTo(new[] { 1.5f, 2.25f }));
        Assert.That(generated[1].Vector.ToArray(), Is.EqualTo(new[] { -3f, 4f }));
    }

    [Test]
    public void ToGeneratedEmbeddings_NullRowWithoutPolicy_Throws()
    {
        var frame = VectorFrame([1f, 2f], null);

        var ex = Assert.Throws<InvalidOperationException>(() => frame.ToGeneratedEmbeddings(Embedding));

        Assert.That(ex!.Message, Does.Contain("Row 1"));
        Assert.That(ex.Message, Does.Contain("nullReplacement"));
    }

    [Test]
    public void ToGeneratedEmbeddings_NullRowWithPolicy_EmitsReplacement()
    {
        var frame = VectorFrame([1f, 2f], null);

        int? seenRow = null;
        int? seenDimension = null;
        var generated = frame.ToGeneratedEmbeddings(Embedding, (row, dimension) =>
        {
            seenRow = row;
            seenDimension = dimension;
            return [9f, 8f];
        });

        Assert.That(generated.Count, Is.EqualTo(2));
        Assert.That(generated[0].Vector.ToArray(), Is.EqualTo(new[] { 1f, 2f }));
        Assert.That(generated[1].Vector.ToArray(), Is.EqualTo(new[] { 9f, 8f }));
        Assert.That(seenRow, Is.EqualTo(1));
        Assert.That(seenDimension, Is.EqualTo(2));
    }

    [Test]
    public void ToGeneratedEmbeddings_PolicyReturnsWrongLength_Throws()
    {
        var frame = VectorFrame([1f, 2f], null);

        var ex = Assert.Throws<InvalidOperationException>(() =>
            frame.ToGeneratedEmbeddings(Embedding, (_, _) => [1f]));

        Assert.That(ex!.Message, Does.Contain("length 1"));
        Assert.That(ex.Message, Does.Contain("expected 2"));
    }

    [Test]
    public void ToGeneratedEmbeddings_PolicyNotInvokedWhenColumnHasNoNulls()
    {
        var frame = VectorFrame([1f, 2f], [3f, 4f]);

        var invoked = false;
        _ = frame.ToGeneratedEmbeddings(Embedding, (_, dimension) =>
        {
            invoked = true;
            return new float[dimension];
        });

        Assert.That(invoked, Is.False);
    }

    [Test]
    public void ToGeneratedEmbeddings_AllNullWithDimensionAndPolicy_Succeeds()
    {
        var frame = VectorFrame(null, null);

        var generated = frame.ToGeneratedEmbeddings(
            Embedding,
            (_, dimension) => new float[dimension],
            dimension: 3);

        Assert.That(generated.Count, Is.EqualTo(2));
        Assert.That(generated[0].Vector.Length, Is.EqualTo(3));
    }

    [Test]
    public void ToGeneratedEmbeddings_AllNullWithoutDimension_Throws()
    {
        var frame = VectorFrame(null, null);

        var ex = Assert.Throws<InvalidOperationException>(() =>
            frame.ToGeneratedEmbeddings(Embedding, (_, dimension) => new float[dimension]));

        Assert.That(ex!.Message, Does.Contain("dimension"));
    }

    // ── scoring: parity with the scalar reference ───────────────────

    [Test]
    public void RowCosineSimilarity_MatchesScalarTensorPrimitivesExactly()
    {
        var frame = VectorFrame([1f, 0f], [0f, 1f], [3f, 4f]);
        var query = new[] { 1f, 0f };

        var scores = frame.RowCosineSimilarity(Embedding, query);

        Assert.That(scores.Length, Is.EqualTo(3));
        for (var row = 0; row < 3; row++)
            Assert.That(
                scores[row],
                Is.EqualTo(TensorPrimitives.CosineSimilarity(query, Vectors(frame)[row])),
                $"row {row}");
    }

    [Test]
    public void RowDot_MatchesScalarTensorPrimitivesExactly()
    {
        var frame = VectorFrame([1f, 0f], [0f, 1f], [3f, 4f]);
        var query = new[] { 2f, 5f };

        var scores = frame.RowDot(Embedding, query);

        for (var row = 0; row < 3; row++)
            Assert.That(
                scores[row],
                Is.EqualTo(TensorPrimitives.Dot(query, Vectors(frame)[row])),
                $"row {row}");
    }

    [Test]
    public void RowCosineSimilarity_SeriesQuery_DelegatesWithLabels()
    {
        var frame = VectorFrame([1f, 0f], [0f, 1f]);
        var query = NivaraSeries<float>.Create([1f, 0f]);
        var labels = NivaraColumn<object>.CreateForReferenceType(["chunk-0", "chunk-1"]);

        var scores = frame.RowCosineSimilarity(Embedding, query, labels);

        var top = scores.TopKDescending(1);
        Assert.That(top, Has.Length.EqualTo(1));
        Assert.That(top[0].Label, Is.EqualTo("chunk-0"));
    }

    [Test]
    public void RowCosineSimilarity_PositionalLabels_AreRecoverableAsRowIndices()
    {
        var frame = VectorFrame([1f, 0f], [0f, 1f], [3f, 4f]);
        var query = new[] { 1f, 0f };

        var top = frame.RowCosineSimilarity(Embedding, query).TopKDescending(2);

        Assert.That(top, Has.Length.EqualTo(2));
        foreach (var (label, _) in top)
            Assert.That(int.TryParse(label, out _), Is.True, $"label '{label}' must parse as a row index");
    }

    // ── scoring: null semantics ─────────────────────────────────────

    [Test]
    public void RowCosineSimilarity_NullEmbeddingRow_ScoresNull()
    {
        var frame = VectorFrame([1f, 0f], null, [1f, 0f]);

        var scores = frame.RowCosineSimilarity(Embedding, new[] { 1f, 0f });

        Assert.That(scores.IsNull(1), Is.True);
        Assert.That(scores.IsNull(0), Is.False);
        Assert.That(scores.IsNull(2), Is.False);
    }

    [Test]
    public void RowCosineSimilarity_NullQueryElement_ScoresAllNull()
    {
        var frame = VectorFrame([1f, 0f], [0f, 1f]);
        var query = new NivaraSeries<float>(NivaraColumnFactory.CreateFromNullable(new float?[] { 1f, null }));

        var scores = frame.RowCosineSimilarity(Embedding, query);

        Assert.That(scores.IsNull(0), Is.True);
        Assert.That(scores.IsNull(1), Is.True);
    }

    [Test]
    public void RowCosineSimilarity_NullArrayQuery_ScoresAllNull()
    {
        var frame = VectorFrame([1f, 0f], [0f, 1f]);

        var scores = frame.RowCosineSimilarity(Embedding, (float[]?)null);

        Assert.That(scores.Length, Is.EqualTo(2));
        Assert.That(scores.IsNull(0), Is.True);
        Assert.That(scores.IsNull(1), Is.True);
    }

    [Test]
    public void RowDot_NullEmbeddingRow_ScoresNull()
    {
        var frame = VectorFrame([1f, 2f], null);

        var scores = frame.RowDot(Embedding, new[] { 1f, 1f });

        Assert.That(scores.IsNull(1), Is.True);
        Assert.That(scores[0], Is.EqualTo(3f));
    }

    [Test]
    public void RowCosineSimilarity_QueryLengthMismatch_Throws()
    {
        var frame = VectorFrame([1f, 0f]);

        var ex = Assert.Throws<ArgumentException>(() =>
            frame.RowCosineSimilarity(Embedding, new[] { 1f, 0f, 0f }));

        Assert.That(ex!.Message, Does.Contain("must match the embedding dimension"));
    }

    [Test]
    public void RowCosineSimilarity_AllNullColumn_UsesQueryDimensionAndScoresAllNull()
    {
        var frame = VectorFrame(null, null);

        var scores = frame.RowCosineSimilarity(Embedding, new[] { 1f, 0f });

        Assert.That(scores.Length, Is.EqualTo(2));
        Assert.That(scores.IsNull(0), Is.True);
        Assert.That(scores.IsNull(1), Is.True);
    }

    [Test]
    public void RowCosineSimilarity_NullArrayQueryAndAllNullColumn_ScoresAllNull()
    {
        var frame = VectorFrame([null]);

        var scores = frame.RowCosineSimilarity(Embedding, (float[]?)null);

        Assert.That(scores.Length, Is.EqualTo(1));
        Assert.That(scores.IsNull(0), Is.True);
    }

    [Test]
    public void RowCosineSimilarity_EmptyQueryAndAllNullColumn_Throws()
    {
        var frame = VectorFrame([null]);

        var ex = Assert.Throws<ArgumentException>(() =>
            frame.RowCosineSimilarity(Embedding, Array.Empty<float>()));

        Assert.That(ex!.Message, Does.Contain("must be positive"));
    }

    // ── the access-control path #912 depends on ─────────────────────

    [Test]
    public void FilterByMask_OnEmbeddingFrame_PreservesVectorsAndNulls()
    {
        var frame = VectorFrame([1f, 2f], null, [5f, 6f]);
        var mask = NivaraColumn<bool>.Create([false, true, true]);

        var filtered = frame.FilterByMask(mask);

        var column = Vectors(filtered);
        Assert.That(column.Length, Is.EqualTo(2));
        Assert.That(column.IsNull(0), Is.True);
        Assert.That(column[1], Is.EqualTo(new[] { 5f, 6f }));

        var scores = filtered.RowCosineSimilarity(Embedding, new[] { 1f, 0f });
        Assert.That(scores.IsNull(0), Is.True);
        Assert.That(scores.IsNull(1), Is.False);
    }

    // ── generate → frame ────────────────────────────────────────────

    [Test]
    public async Task GenerateFrameAsync_LandsGeneratedVectors()
    {
        var generator = new NivaraEmbeddingGenerator<string>(
            text => [(float)text.Length, 1f],
            embeddingDimension: 2);

        var frame = await generator.GenerateFrameAsync(["a", "bbb"]);

        Assert.That(frame.RowCount, Is.EqualTo(2));
        Assert.That(Vectors(frame)[0], Is.EqualTo(new[] { 1f, 1f }));
        Assert.That(Vectors(frame)[1], Is.EqualTo(new[] { 3f, 1f }));
    }
}
