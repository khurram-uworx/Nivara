using Nivara.Operations;
using NUnit.Framework;

namespace Nivara.Tests;

/// <summary>
/// Integration tests for sorting operations demonstrating end-to-end functionality
/// </summary>
[TestFixture]
public class SortingIntegrationTests
{
    [Test]
    public void SortingWorkflow_CompleteExample_ShouldWorkEndToEnd()
    {
        // Arrange - Create a sample employee dataset
        var names = NivaraColumn<string>.CreateForReferenceType(new[]
        {
            "Alice Johnson", "Bob Smith", "Charlie Brown", "Diana Prince", "Eve Adams"
        });
        var departments = NivaraColumn<string>.CreateForReferenceType(new[]
        {
            "Engineering", "Sales", "Engineering", "Marketing", "Sales"
        });
        var salaries = NivaraColumn<double>.Create(new[]
        {
            75000.0, 65000.0, 85000.0, 70000.0, 68000.0
        });
        var ages = NivaraColumn<int>.Create(new[]
        {
            28, 35, 42, 31, 29
        });

        var frame = NivaraFrame.Create(
            ("Name", names),
            ("Department", departments),
            ("Salary", salaries),
            ("Age", ages)
        );

        // Act 1: Sort by single column (salary descending)
        var sortedBySalary = frame.AsQueryFrame()
            .Sort("Salary", SortDirection.Descending)
            .Collect();

        // Assert 1: Verify salary sorting
        var sortedSalaries = sortedBySalary.GetColumn<double>("Salary");
        var sortedNames = sortedBySalary.GetColumn<string>("Name");

        Assert.That(sortedSalaries.ToArray(), Is.EqualTo(new[] { 85000.0, 75000.0, 70000.0, 68000.0, 65000.0 }));
        Assert.That(sortedNames.ToArray(), Is.EqualTo(new[] { "Charlie Brown", "Alice Johnson", "Diana Prince", "Eve Adams", "Bob Smith" }));

        // Act 2: Multi-column sort (department ascending, then salary descending within department)
        var sortKeys = new[]
        {
            new SortKey("Department", SortDirection.Ascending),
            new SortKey("Salary", SortDirection.Descending)
        };

        var multiSorted = frame.AsQueryFrame()
            .Sort(sortKeys)
            .Collect();

        // Assert 2: Verify multi-column sorting
        var multiSortedDepts = multiSorted.GetColumn<string>("Department");
        var multiSortedSalaries = multiSorted.GetColumn<double>("Salary");
        var multiSortedNames = multiSorted.GetColumn<string>("Name");

        // Expected order: Engineering (Charlie 85000, Alice 75000), Marketing (Diana 70000), Sales (Eve 68000, Bob 65000)
        Assert.That(multiSortedDepts.ToArray(), Is.EqualTo(new[] { "Engineering", "Engineering", "Marketing", "Sales", "Sales" }));
        Assert.That(multiSortedSalaries.ToArray(), Is.EqualTo(new[] { 85000.0, 75000.0, 70000.0, 68000.0, 65000.0 }));
        Assert.That(multiSortedNames.ToArray(), Is.EqualTo(new[] { "Charlie Brown", "Alice Johnson", "Diana Prince", "Eve Adams", "Bob Smith" }));

        // Act 3: Test with null values
        var scoresWithNulls = NivaraColumn<int>.CreateFromNullable(new int?[] { 85, null, 92, 78, null });
        var frameWithNulls = frame.WithColumn("Score", scoresWithNulls);

        var sortedWithNullsFirst = frameWithNulls.AsQueryFrame()
            .Sort("Score", SortDirection.Ascending, NullOrdering.NullsFirst)
            .Collect();

        // Assert 3: Verify null handling
        var sortedScores = sortedWithNullsFirst.GetColumn<int>("Score");
        var sortedNamesWithNulls = sortedWithNullsFirst.GetColumn<string>("Name");

        // Expected order: nulls first, then 78, 85, 92
        Assert.That(sortedScores.GetValue(0), Is.Null); // Bob Smith (null)
        Assert.That(sortedScores.GetValue(1), Is.Null); // Eve Adams (null)
        Assert.That(sortedScores.GetValue(2), Is.EqualTo(78)); // Diana Prince
        Assert.That(sortedScores.GetValue(3), Is.EqualTo(85)); // Alice Johnson
        Assert.That(sortedScores.GetValue(4), Is.EqualTo(92)); // Charlie Brown

        Assert.That(sortedNamesWithNulls.ToArray(), Is.EqualTo(new[] { "Bob Smith", "Eve Adams", "Diana Prince", "Alice Johnson", "Charlie Brown" }));
    }

    [Test]
    public void SortingWithFiltering_CombinedOperations_ShouldWorkCorrectly()
    {
        // Arrange
        var names = NivaraColumn<string>.CreateForReferenceType(new[]
        {
            "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"
        });
        var ages = NivaraColumn<int>.Create(new[] { 25, 35, 45, 30, 28, 50 });
        var salaries = NivaraColumn<double>.Create(new[] { 50000.0, 60000.0, 80000.0, 55000.0, 52000.0, 90000.0 });

        var frame = NivaraFrame.Create(
            ("Name", names),
            ("Age", ages),
            ("Salary", salaries)
        );

        // Act: Filter employees over 30, then sort by salary descending
        var mask = NivaraColumn<bool>.Create(new[] { false, true, true, false, false, true }); // Bob, Charlie, Frank

        var result = frame.FilterByMask(mask)
            .AsQueryFrame()
            .Sort("Salary", SortDirection.Descending)
            .Collect();

        // Assert
        var resultNames = result.GetColumn<string>("Name");
        var resultSalaries = result.GetColumn<double>("Salary");

        Assert.That(result.RowCount, Is.EqualTo(3));
        Assert.That(resultNames.ToArray(), Is.EqualTo(new[] { "Frank", "Charlie", "Bob" }));
        Assert.That(resultSalaries.ToArray(), Is.EqualTo(new[] { 90000.0, 80000.0, 60000.0 }));
    }

    [Test]
    public void SortingWithSlicing_CombinedOperations_ShouldWorkCorrectly()
    {
        // Arrange
        var numbers = NivaraColumn<int>.Create(new[] { 5, 2, 8, 1, 9, 3, 7, 4, 6 });
        var letters = NivaraColumn<string>.CreateForReferenceType(new[] { "e", "b", "h", "a", "i", "c", "g", "d", "f" });

        var frame = NivaraFrame.Create(
            ("Numbers", numbers),
            ("Letters", letters)
        );

        // Act: Sort by numbers, then take middle 3 elements (skip 3, take 3)
        var result = frame.AsQueryFrame()
            .Sort("Numbers", SortDirection.Ascending)
            .Collect()
            .Skip(3)
            .Take(3);

        // Assert
        var resultNumbers = result.GetColumn<int>("Numbers");
        var resultLetters = result.GetColumn<string>("Letters");

        // After sorting: 1,a -> 2,b -> 3,c -> 4,d -> 5,e -> 6,f -> 7,g -> 8,h -> 9,i
        // Skip 3 (1,2,3), take 3 (4,5,6)
        Assert.That(result.RowCount, Is.EqualTo(3));
        Assert.That(resultNumbers.ToArray(), Is.EqualTo(new[] { 4, 5, 6 }));
        Assert.That(resultLetters.ToArray(), Is.EqualTo(new[] { "d", "e", "f" }));
    }

    [Test]
    public void ReorderByIndices_DirectUsage_ShouldWorkCorrectly()
    {
        // Arrange
        var data = NivaraFrame.Create(
            ("ID", NivaraColumn<int>.Create(new[] { 100, 200, 300, 400 })),
            ("Value", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "B", "C", "D" }))
        );

        // Act: Reverse the order using indices
        var reverseIndices = new[] { 3, 2, 1, 0 };
        var reversed = data.ReorderByIndices(reverseIndices);

        // Assert
        var reversedIds = reversed.GetColumn<int>("ID");
        var reversedValues = reversed.GetColumn<string>("Value");

        Assert.That(reversedIds.ToArray(), Is.EqualTo(new[] { 400, 300, 200, 100 }));
        Assert.That(reversedValues.ToArray(), Is.EqualTo(new[] { "D", "C", "B", "A" }));
    }

    [Test]
    public void SortingPerformance_LargeDataset_ShouldCompleteReasonably()
    {
        // Arrange - Create a larger dataset for performance testing
        const int rowCount = 10000;
        var random = new Random(42); // Fixed seed for reproducible results

        var ids = NivaraColumn<int>.Create(Enumerable.Range(0, rowCount).ToArray());
        var randomValues = NivaraColumn<double>.Create(
            Enumerable.Range(0, rowCount).Select(_ => random.NextDouble() * 1000).ToArray()
        );
        var categories = NivaraColumn<string>.CreateForReferenceType(
            Enumerable.Range(0, rowCount).Select(i => $"Category{i % 10}").ToArray()
        );

        var largeFrame = NivaraFrame.Create(
            ("ID", ids),
            ("Value", randomValues),
            ("Category", categories)
        );

        // Act & Assert - Should complete without timeout
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        var sorted = largeFrame.AsQueryFrame()
            .Sort("Value", SortDirection.Descending)
            .Collect();

        stopwatch.Stop();

        // Verify sorting worked correctly
        var sortedValues = sorted.GetColumn<double>("Value");
        Assert.That(sorted.RowCount, Is.EqualTo(rowCount));

        // Check that values are in descending order
        for (int i = 0; i < sortedValues.Length - 1; i++)
        {
            Assert.That(sortedValues[i], Is.GreaterThanOrEqualTo(sortedValues[i + 1]),
                $"Values not in descending order at index {i}: {sortedValues[i]} < {sortedValues[i + 1]}");
        }

        // Performance assertion - should complete in reasonable time (adjust as needed)
        Assert.That(stopwatch.ElapsedMilliseconds, Is.LessThan(60000),
            $"Sorting {rowCount} rows took {stopwatch.ElapsedMilliseconds}ms, which may be too slow");

        Console.WriteLine($"Sorted {rowCount} rows in {stopwatch.ElapsedMilliseconds}ms");
    }
}