using Nivara.Exceptions;
using Nivara.Execution;
using Nivara.Operations;
using NUnit.Framework;

namespace Nivara.Tests.Execution;

[TestFixture]
public class ParallelExecutionHelperTests
{
    [TestCase(0, 100)]
    [TestCase(-5, 100)]
    [TestCase(1, 100)]
    public void CalculateOptimalChunkSize_NonPositiveTotal_ReturnsMinChunkSize(int total, int expected)
    {
        var result = ParallelExecutionHelper.CalculateOptimalChunkSize(total, 4, minChunkSize: 100, maxChunkSize: 10000);

        Assert.That(result, Is.EqualTo(expected));
    }

    [Test]
    public void CalculateOptimalChunkSize_SmallItems_ReturnsMinChunkSize()
    {
        var result = ParallelExecutionHelper.CalculateOptimalChunkSize(50, 4, minChunkSize: 100, maxChunkSize: 10000);

        Assert.That(result, Is.EqualTo(100));
    }

    [Test]
    public void CalculateOptimalChunkSize_LargeItems_CappedAtMaxChunkSize()
    {
        var result = ParallelExecutionHelper.CalculateOptimalChunkSize(1_000_000, 4, minChunkSize: 100, maxChunkSize: 10000);

        Assert.That(result, Is.EqualTo(10000));
    }

    [Test]
    public void CalculateOptimalChunkSize_RespectsCustomMin()
    {
        var result = ParallelExecutionHelper.CalculateOptimalChunkSize(500, 4, minChunkSize: 50, maxChunkSize: 10000);

        Assert.That(result, Is.EqualTo(50));
    }

    [Test]
    public void CalculateOptimalChunkSize_RespectsCustomMax()
    {
        var result = ParallelExecutionHelper.CalculateOptimalChunkSize(500_000, 4, minChunkSize: 100, maxChunkSize: 5000);

        Assert.That(result, Is.EqualTo(5000));
    }

    [Test]
    public void ShouldUseParallelProcessing_DopLessOrEqualOne_ReturnsFalse()
    {
        Assert.That(ParallelExecutionHelper.ShouldUseParallelProcessing(2000, 0), Is.False);
        Assert.That(ParallelExecutionHelper.ShouldUseParallelProcessing(2000, 1), Is.False);
    }

    [Test]
    public void ShouldUseParallelProcessing_ItemCountBelowThreshold_ReturnsFalse()
    {
        Assert.That(ParallelExecutionHelper.ShouldUseParallelProcessing(10, 4), Is.False);
    }

    [Test]
    public void ShouldUseParallelProcessing_ValidConditions_ReturnsTrue()
    {
        Assert.That(ParallelExecutionHelper.ShouldUseParallelProcessing(2000, 4), Is.True);
    }

    [Test]
    public void GetRecommendedParallelism_NeverExceedsProcessorCountTimesTwo()
    {
        var result = ParallelExecutionHelper.GetRecommendedParallelism(int.MaxValue);

        Assert.That(result, Is.LessThanOrEqualTo(Environment.ProcessorCount * 2));
    }

    [Test]
    public void GetRecommendedParallelism_ReturnsRequestedValueWhenWithinBounds()
    {
        var result = ParallelExecutionHelper.GetRecommendedParallelism(2);

        Assert.That(result, Is.EqualTo(2));
    }

    [Test]
    public void CreateChunkRanges_ExactlyDivisible_AllChunksSameSize()
    {
        var ranges = ParallelExecutionHelper.CreateChunkRanges(100, 20);

        Assert.That(ranges.Count, Is.EqualTo(5));
        foreach (var (start, length) in ranges)
            Assert.That(length, Is.EqualTo(20));
        Assert.That(ranges[0].Start, Is.EqualTo(0));
        Assert.That(ranges[4].Start, Is.EqualTo(80));
    }

    [Test]
    public void CreateChunkRanges_PartialFinalChunk_LastChunkSmaller()
    {
        var ranges = ParallelExecutionHelper.CreateChunkRanges(105, 20);

        Assert.That(ranges.Count, Is.EqualTo(6));
        Assert.That(ranges[0].Length, Is.EqualTo(20));
        Assert.That(ranges[5].Length, Is.EqualTo(5));
    }

    [Test]
    public void CreateChunkRanges_ZeroRows_ReturnsEmpty()
    {
        var ranges = ParallelExecutionHelper.CreateChunkRanges(0, 20);

        Assert.That(ranges, Is.Empty);
    }

    [Test]
    public void ConcatenateColumnDictionaries_SingleChunk_ReturnsAsIs()
    {
        var chunk = new Dictionary<string, IColumn>
        { ["A"] = NivaraColumn<int>.Create(new[] { 1, 2 }) };

        var result = ParallelExecutionHelper.ConcatenateColumnDictionaries(
            new[] { chunk });

        Assert.That(result, Is.SameAs(chunk));
    }

    [Test]
    public void ConcatenateColumnDictionaries_TwoChunks_ConcatenatesVertically()
    {
        var chunks = new[]
        {
            new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(new[] { 1, 2 }) },
            new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(new[] { 3, 4 }) },
        };

        var result = ParallelExecutionHelper.ConcatenateColumnDictionaries(chunks);

        Assert.That(result["A"].Length, Is.EqualTo(4));
    }

    [Test]
    public void ConcatenateColumnDictionaries_MismatchedColumns_IncludesAll()
    {
        var chunks = new[]
        {
            new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(new[] { 1 }) },
            new Dictionary<string, IColumn> { ["B"] = NivaraColumn<int>.Create(new[] { 2 }) },
        };

        var result = ParallelExecutionHelper.ConcatenateColumnDictionaries(chunks);

        Assert.That(result.ContainsKey("A"), Is.True);
        Assert.That(result.ContainsKey("B"), Is.True);
    }

    [Test]
    public void ConcatenateColumnDictionaries_AllEmpty_HandledGracefully()
    {
        var chunks = new[]
        {
            new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(Array.Empty<int>()) },
            new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(Array.Empty<int>()) },
        };

        var result = ParallelExecutionHelper.ConcatenateColumnDictionaries(chunks);

        Assert.That(result, Is.Not.Null);
    }

    [Test]
    public void ConcatenateColumnDictionaries_ThreeOrMoreChunks_Works()
    {
        var chunks = new[]
        {
            new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(new[] { 1 }) },
            new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(new[] { 2 }) },
            new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(new[] { 3 }) },
        };

        var result = ParallelExecutionHelper.ConcatenateColumnDictionaries(chunks);

        Assert.That(result["A"].Length, Is.EqualTo(3));
    }

    [Test]
    public void MergeGroupByDictionaries_SingleMap_ReturnsCopy()
    {
        var key = new GroupKey(new[] { "a" }, new object[] { 1 });
        var partial = new Dictionary<GroupKey, List<int>>
        {
            [key] = new List<int> { 0, 1 }
        };

        var result = ParallelExecutionHelper.MergeGroupByDictionaries(
            new[] { partial });

        Assert.That(result.Count, Is.EqualTo(1));
        Assert.That(result[key], Is.EquivalentTo(new[] { 0, 1 }));
    }

    [Test]
    public void MergeGroupByDictionaries_OverlappingKeys_MergesCorrectly()
    {
        var key = new GroupKey(new[] { "a" }, new object[] { 1 });
        var partials = new[]
        {
            new Dictionary<GroupKey, List<int>> { [key] = new List<int> { 0, 1 } },
            new Dictionary<GroupKey, List<int>> { [key] = new List<int> { 2, 3 } },
        };

        var result = ParallelExecutionHelper.MergeGroupByDictionaries(partials);

        Assert.That(result[key], Is.EquivalentTo(new[] { 0, 1, 2, 3 }));
    }

    [Test]
    public void MergeGroupByDictionaries_EmptyInput_ReturnsEmpty()
    {
        var result = ParallelExecutionHelper.MergeGroupByDictionaries(
            Array.Empty<Dictionary<GroupKey, List<int>>>());

        Assert.That(result, Is.Empty);
    }

    [Test]
    public void MergeJoinHashMaps_SingleMap_ReturnsCopy()
    {
        var key = new CompositeKey(new IColumn[]
        {
            NivaraColumn<int>.Create(new[] { 1 })
        });
        var partial = new Dictionary<CompositeKey, List<int>>
        {
            [key] = new List<int> { 0 }
        };

        var result = ParallelExecutionHelper.MergeJoinHashMaps(
            new[] { partial });

        Assert.That(result.Count, Is.EqualTo(1));
    }

    [Test]
    public void MergeJoinHashMaps_OverlappingKeys_MergesCorrectly()
    {
        var key = new CompositeKey(new IColumn[]
        {
            NivaraColumn<int>.Create(new[] { 1 })
        });
        var partials = new[]
        {
            new Dictionary<CompositeKey, List<int>> { [key] = new List<int> { 0 } },
            new Dictionary<CompositeKey, List<int>> { [key] = new List<int> { 1 } },
        };

        var result = ParallelExecutionHelper.MergeJoinHashMaps(partials);

        Assert.That(result[key], Is.EquivalentTo(new[] { 0, 1 }));
    }

    [Test]
    public void MergeJoinHashMaps_EmptyInput_ReturnsEmpty()
    {
        var result = ParallelExecutionHelper.MergeJoinHashMaps(
            Array.Empty<Dictionary<CompositeKey, List<int>>>());

        Assert.That(result, Is.Empty);
    }

    [Test]
    public void MergeSortedChunks_SingleChunk_ProducesSortedResult()
    {
        var values = new[] { 3, 1, 2 };
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<int>.Create(values)
        };
        var ranges = new List<(int, int)> { (0, 3) };
        var sortedLocal = new[] { new[] { 1, 2, 0 } };
        var chunkStarts = new[] { 0 };
        var sortKeys = new[] { new SortKey("A", SortDirection.Ascending) };

        var result = ParallelExecutionHelper.MergeSortedChunks(
            input, sortedLocal, chunkStarts, ranges, sortKeys);

        var col = (NivaraColumn<int>)input["A"];
        var sortedValues = result.Select(i => col[i]).ToArray();
        Assert.That(sortedValues, Is.EqualTo(new[] { 1, 2, 3 }));
    }

    [Test]
    public void MergeSortedChunks_TwoChunks_MergesCorrectly()
    {
        var values = new[] { 1, 4, 2, 3 };
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<int>.Create(values)
        };
        var ranges = new List<(int, int)> { (0, 2), (2, 2) };
        var sortedLocal = new[] { new[] { 0, 1 }, new[] { 0, 1 } };
        var chunkStarts = new[] { 0, 2 };
        var sortKeys = new[] { new SortKey("A", SortDirection.Ascending) };

        var result = ParallelExecutionHelper.MergeSortedChunks(
            input, sortedLocal, chunkStarts, ranges, sortKeys);

        var col = (NivaraColumn<int>)input["A"];
        var sortedValues = result.Select(i => col[i]).ToArray();
        Assert.That(sortedValues, Is.EqualTo(new[] { 1, 2, 3, 4 }));
    }

    [Test]
    public void MergeSortedChunks_TieBreaking_StableOrderPreserved()
    {
        var values = new[] { 1, 1, 2, 2 };
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<int>.Create(values)
        };
        var ranges = new List<(int, int)> { (0, 2), (2, 2) };
        var sortedLocal = new[] { new[] { 0, 1 }, new[] { 0, 1 } };
        var chunkStarts = new[] { 0, 2 };
        var sortKeys = new[] { new SortKey("A", SortDirection.Ascending) };

        var result = ParallelExecutionHelper.MergeSortedChunks(
            input, sortedLocal, chunkStarts, ranges, sortKeys);

        Assert.That(result.Length, Is.EqualTo(4));
        var col = (NivaraColumn<int>)input["A"];
        var sortedValues = result.Select(i => col[i]).ToArray();
        Assert.That(sortedValues, Is.EqualTo(new[] { 1, 1, 2, 2 }));
    }

    [Test]
    public void ValidateParallelConfiguration_DopZero_Throws()
    {
        var ex = Assert.Throws<QueryExecutionException>(() =>
            ParallelExecutionHelper.ValidateParallelConfiguration(0));
        Assert.That(ex!.Message, Does.Contain("greater than zero"));
    }

    [Test]
    public void ValidateParallelConfiguration_DopNegative_Throws()
    {
        Assert.Throws<QueryExecutionException>(() =>
            ParallelExecutionHelper.ValidateParallelConfiguration(-1));
    }

    [Test]
    public void ValidateParallelConfiguration_ExcessiveDop_Throws()
    {
        var ex = Assert.Throws<QueryExecutionException>(() =>
            ParallelExecutionHelper.ValidateParallelConfiguration(Environment.ProcessorCount * 4 + 1));
        Assert.That(ex!.Message, Does.Contain("too high"));
    }

    [Test]
    public void ValidateParallelConfiguration_ValidDop_Passes()
    {
        Assert.DoesNotThrow(() =>
            ParallelExecutionHelper.ValidateParallelConfiguration(Environment.ProcessorCount));
    }

    [Test]
    public void CreateRowSubset_ExtractsCorrectRange()
    {
        var source = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<int>.Create(new[] { 0, 1, 2, 3, 4 })
        };

        var subset = ParallelExecutionHelper.CreateRowSubset(source, 1, 3);

        var col = (NivaraColumn<int>)subset["A"];
        Assert.That(col.Length, Is.EqualTo(3));
        Assert.That(col[0], Is.EqualTo(1));
        Assert.That(col[2], Is.EqualTo(3));
    }

    [Test]
    public void SliceColumn_DelegatesToColumnSlice()
    {
        var col = NivaraColumn<int>.Create(new[] { 0, 1, 2, 3, 4 });

        var sliced = ParallelExecutionHelper.SliceColumn(col, 2, 2);

        Assert.That(sliced.Length, Is.EqualTo(2));
    }

    [Test]
    public void CreateRowSubset_NegativeStartRow_ThrowsArgumentOutOfRangeException()
    {
        var source = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<int>.Create(new[] { 0, 1, 2, 3, 4 })
        };

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ParallelExecutionHelper.CreateRowSubset(source, -1, 3));
    }

    [Test]
    public void CreateRowSubset_StartRowExceedsTotal_ThrowsArgumentOutOfRangeException()
    {
        var source = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<int>.Create(new[] { 0, 1, 2, 3, 4 })
        };

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ParallelExecutionHelper.CreateRowSubset(source, 10, 3));
    }

    [Test]
    public void SliceColumn_NegativeStartRow_ThrowsArgumentOutOfRangeException()
    {
        var col = NivaraColumn<int>.Create(new[] { 0, 1, 2, 3, 4 });

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ParallelExecutionHelper.SliceColumn(col, -1, 2));
    }

    [Test]
    public void SliceColumn_LengthExceedsData_ThrowsArgumentOutOfRangeException()
    {
        var col = NivaraColumn<int>.Create(new[] { 0, 1, 2, 3, 4 });

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ParallelExecutionHelper.SliceColumn(col, 3, 10));
    }

}
