using NUnit.Framework;
using Nivara;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Nivara.Tests.Operations;

[TestFixture]
public class GroupedDataTests
{
    [Test]
    public void Constructor_WithValidParameters_CreatesGroupedData()
    {
        // Arrange
        var groups = new Dictionary<GroupKey, List<int>>
        {
            [new GroupKey("Alice")] = new List<int> { 0, 2 },
            [new GroupKey("Bob")] = new List<int> { 1, 3 }
        };

        var keyColumnNames = new[] { "Name" };
        var sourceColumns = new Dictionary<string, IColumn>
        {
            ["Name"] = NivaraColumn<string>.Create(new[] { "Alice", "Bob", "Alice", "Bob" })
        };

        // Act
        var groupedData = new GroupedData(groups, keyColumnNames, sourceColumns);

        // Assert
        Assert.That(groupedData.GroupCount, Is.EqualTo(2));
        Assert.That(groupedData.KeyColumnNames, Has.Count.EqualTo(1));
        Assert.That(groupedData.KeyColumnNames[0], Is.EqualTo("Name"));
        Assert.That(groupedData.SourceColumns, Has.Count.EqualTo(1));
    }

    [Test]
    public void GetGroupIndices_WithExistingKey_ReturnsCorrectIndices()
    {
        // Arrange
        var aliceKey = new GroupKey("Alice");
        var bobKey = new GroupKey("Bob");

        var groups = new Dictionary<GroupKey, List<int>>
        {
            [aliceKey] = new List<int> { 0, 2, 4 },
            [bobKey] = new List<int> { 1, 3 }
        };

        var keyColumnNames = new[] { "Name" };
        var sourceColumns = new Dictionary<string, IColumn>
        {
            ["Name"] = NivaraColumn<string>.Create(new[] { "Alice", "Bob", "Alice", "Bob", "Alice" })
        };

        var groupedData = new GroupedData(groups, keyColumnNames, sourceColumns);

        // Act
        var aliceIndices = groupedData.GetGroupIndices(aliceKey);
        var bobIndices = groupedData.GetGroupIndices(bobKey);

        // Assert
        Assert.That(aliceIndices, Has.Count.EqualTo(3));
        Assert.That(aliceIndices, Is.EqualTo(new[] { 0, 2, 4 }));
        Assert.That(bobIndices, Has.Count.EqualTo(2));
        Assert.That(bobIndices, Is.EqualTo(new[] { 1, 3 }));
    }

    [Test]
    public void GetGroupIndices_WithNonExistentKey_ReturnsEmptyList()
    {
        // Arrange
        var groups = new Dictionary<GroupKey, List<int>>
        {
            [new GroupKey("Alice")] = new List<int> { 0, 2 }
        };

        var keyColumnNames = new[] { "Name" };
        var sourceColumns = new Dictionary<string, IColumn>
        {
            ["Name"] = NivaraColumn<string>.Create(new[] { "Alice", "Bob", "Alice" })
        };

        var groupedData = new GroupedData(groups, keyColumnNames, sourceColumns);
        var nonExistentKey = new GroupKey("Charlie");

        // Act
        var indices = groupedData.GetGroupIndices(nonExistentKey);

        // Assert
        Assert.That(indices, Is.Empty);
    }

    [Test]
    public void GetAllGroups_ReturnsAllGroupsWithIndices()
    {
        // Arrange
        var aliceKey = new GroupKey("Alice");
        var bobKey = new GroupKey("Bob");

        var groups = new Dictionary<GroupKey, List<int>>
        {
            [aliceKey] = new List<int> { 0, 2 },
            [bobKey] = new List<int> { 1, 3 }
        };

        var keyColumnNames = new[] { "Name" };
        var sourceColumns = new Dictionary<string, IColumn>
        {
            ["Name"] = NivaraColumn<string>.Create(new[] { "Alice", "Bob", "Alice", "Bob" })
        };

        var groupedData = new GroupedData(groups, keyColumnNames, sourceColumns);

        // Act
        var allGroups = groupedData.GetAllGroups().ToList();

        // Assert
        Assert.That(allGroups, Has.Count.EqualTo(2));

        var aliceGroup = allGroups.FirstOrDefault(g => g.Key.Equals(aliceKey));
        var bobGroup = allGroups.FirstOrDefault(g => g.Key.Equals(bobKey));

        Assert.That(aliceGroup.Key, Is.Not.Null);
        Assert.That(aliceGroup.Indices, Is.EqualTo(new[] { 0, 2 }));

        Assert.That(bobGroup.Key, Is.Not.Null);
        Assert.That(bobGroup.Indices, Is.EqualTo(new[] { 1, 3 }));
    }

    [Test]
    public void GroupKeys_ReturnsAllKeys()
    {
        // Arrange
        var aliceKey = new GroupKey("Alice");
        var bobKey = new GroupKey("Bob");

        var groups = new Dictionary<GroupKey, List<int>>
        {
            [aliceKey] = new List<int> { 0, 2 },
            [bobKey] = new List<int> { 1, 3 }
        };

        var keyColumnNames = new[] { "Name" };
        var sourceColumns = new Dictionary<string, IColumn>
        {
            ["Name"] = NivaraColumn<string>.Create(new[] { "Alice", "Bob", "Alice", "Bob" })
        };

        var groupedData = new GroupedData(groups, keyColumnNames, sourceColumns);

        // Act
        var keys = groupedData.GroupKeys.ToList();

        // Assert
        Assert.That(keys, Has.Count.EqualTo(2));
        Assert.That(keys, Contains.Item(aliceKey));
        Assert.That(keys, Contains.Item(bobKey));
    }
}

[TestFixture]
public class GroupKeyTests
{
    [Test]
    public void Constructor_WithValidValues_CreatesGroupKey()
    {
        // Arrange & Act
        var key = new GroupKey("Alice", 25);

        // Assert
        Assert.That(key.Values, Has.Count.EqualTo(2));
        Assert.That(key.Values[0], Is.EqualTo("Alice"));
        Assert.That(key.Values[1], Is.EqualTo(25));
    }

    [Test]
    public void Constructor_WithNullValues_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new GroupKey(null!));
    }

    [Test]
    public void Equals_WithSameValues_ReturnsTrue()
    {
        // Arrange
        var key1 = new GroupKey("Alice", 25);
        var key2 = new GroupKey("Alice", 25);

        // Act & Assert
        Assert.That(key1.Equals(key2), Is.True);
        Assert.That(key1 == key2, Is.False); // Reference equality
        Assert.That(key1.GetHashCode(), Is.EqualTo(key2.GetHashCode()));
    }

    [Test]
    public void Equals_WithDifferentValues_ReturnsFalse()
    {
        // Arrange
        var key1 = new GroupKey("Alice", 25);
        var key2 = new GroupKey("Bob", 30);

        // Act & Assert
        Assert.That(key1.Equals(key2), Is.False);
    }

    [Test]
    public void Equals_WithDifferentLengths_ReturnsFalse()
    {
        // Arrange
        var key1 = new GroupKey("Alice");
        var key2 = new GroupKey("Alice", 25);

        // Act & Assert
        Assert.That(key1.Equals(key2), Is.False);
    }

    [Test]
    public void Equals_WithNullValues_HandlesCorrectly()
    {
        // Arrange
        var key1 = new GroupKey("Alice", null);
        var key2 = new GroupKey("Alice", null);
        var key3 = new GroupKey("Alice", 25);

        // Act & Assert
        Assert.That(key1.Equals(key2), Is.True);
        Assert.That(key1.Equals(key3), Is.False);
    }

    [Test]
    public void Equals_WithNull_ReturnsFalse()
    {
        // Arrange
        var key = new GroupKey("Alice");

        // Act & Assert
        Assert.That(key.Equals(null), Is.False);
    }

    [Test]
    public void ToString_ReturnsFormattedString()
    {
        // Arrange
        var key = new GroupKey("Alice", 25, null);

        // Act
        var result = key.ToString();

        // Assert
        Assert.That(result, Is.EqualTo("(Alice, 25, null)"));
    }

    [Test]
    public void GetHashCode_WithSameValues_ReturnsSameHashCode()
    {
        // Arrange
        var key1 = new GroupKey("Alice", 25);
        var key2 = new GroupKey("Alice", 25);

        // Act
        var hash1 = key1.GetHashCode();
        var hash2 = key2.GetHashCode();

        // Assert
        Assert.That(hash1, Is.EqualTo(hash2));
    }

    [Test]
    public void GetHashCode_WithDifferentValues_ReturnsDifferentHashCodes()
    {
        // Arrange
        var key1 = new GroupKey("Alice", 25);
        var key2 = new GroupKey("Bob", 30);

        // Act
        var hash1 = key1.GetHashCode();
        var hash2 = key2.GetHashCode();

        // Assert
        Assert.That(hash1, Is.Not.EqualTo(hash2));
    }
}