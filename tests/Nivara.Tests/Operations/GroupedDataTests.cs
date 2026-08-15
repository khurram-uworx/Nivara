using NUnit.Framework;

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
            [GroupKey.FromValues(new object[] { "Alice" })] = new List<int> { 0, 2 },
            [GroupKey.FromValues(new object[] { "Bob" })] = new List<int> { 1, 3 }
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
        var aliceKey = GroupKey.FromValues(new object[] { "Alice" });
        var bobKey = GroupKey.FromValues(new object[] { "Bob" });

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
            [GroupKey.FromValues(new object[] { "Alice" })] = new List<int> { 0, 2 }
        };

        var keyColumnNames = new[] { "Name" };
        var sourceColumns = new Dictionary<string, IColumn>
        {
            ["Name"] = NivaraColumn<string>.Create(new[] { "Alice", "Bob", "Alice" })
        };

        var groupedData = new GroupedData(groups, keyColumnNames, sourceColumns);
        var nonExistentKey = GroupKey.FromValues(new object[] { "Charlie" });

        // Act
        var indices = groupedData.GetGroupIndices(nonExistentKey);

        // Assert
        Assert.That(indices, Is.Empty);
    }

    [Test]
    public void GetAllGroups_ReturnsAllGroupsWithIndices()
    {
        // Arrange
        var aliceKey = GroupKey.FromValues(new object[] { "Alice" });
        var bobKey = GroupKey.FromValues(new object[] { "Bob" });

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
        var aliceKey = GroupKey.FromValues(new object[] { "Alice" });
        var bobKey = GroupKey.FromValues(new object[] { "Bob" });

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
        var key = GroupKey.FromValues(new object[] { "Alice", 25 });

        // Assert
        Assert.That(key.Values, Has.Count.EqualTo(2));
        Assert.That(key.Values[0], Is.EqualTo("Alice"));
        Assert.That(key.Values[1], Is.EqualTo(25));
    }

    [Test]
    public void Constructor_WithNullValues_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => GroupKey.FromValues(null!));
    }

    [Test]
    public void Equals_WithSameValues_ReturnsTrue()
    {
        // Arrange
        var key1 = GroupKey.FromValues(new object[] { "Alice", 25 });
        var key2 = GroupKey.FromValues(new object[] { "Alice", 25 });

        // Act & Assert
        Assert.That(key1.Equals(key2), Is.True);
        Assert.That(key1 == key2, Is.False); // Reference equality
        Assert.That(key1.GetHashCode(), Is.EqualTo(key2.GetHashCode()));
    }

    [Test]
    public void Equals_WithDifferentValues_ReturnsFalse()
    {
        // Arrange
        var key1 = GroupKey.FromValues(new object[] { "Alice", 25 });
        var key2 = GroupKey.FromValues(new object[] { "Bob", 30 });

        // Act & Assert
        Assert.That(key1.Equals(key2), Is.False);
    }

    [Test]
    public void Equals_WithDifferentLengths_ReturnsFalse()
    {
        // Arrange
        var key1 = GroupKey.FromValues(new object[] { "Alice" });
        var key2 = GroupKey.FromValues(new object[] { "Alice", 25 });

        // Act & Assert
        Assert.That(key1.Equals(key2), Is.False);
    }

    [Test]
    public void Equals_WithNullValues_HandlesCorrectly()
    {
        // Arrange
        var key1 = GroupKey.FromValues(new object?[] { "Alice", null });
        var key2 = GroupKey.FromValues(new object?[] { "Alice", null });
        var key3 = GroupKey.FromValues(new object[] { "Alice", 25 });

        // Act & Assert
        Assert.That(key1.Equals(key2), Is.True);
        Assert.That(key1.Equals(key3), Is.False);
    }

    [Test]
    public void Equals_WithNull_ReturnsFalse()
    {
        // Arrange
        var key = GroupKey.FromValues(new object[] { "Alice" });

        // Act & Assert
        Assert.That(key.Equals(null), Is.False);
    }

    [Test]
    public void ToString_ReturnsFormattedString()
    {
        // Arrange
        var key = GroupKey.FromValues(new object?[] { "Alice", 25, null });

        // Act
        var result = key.ToString();

        // Assert
        Assert.That(result, Is.EqualTo("(Alice, 25, null)"));
    }

    [Test]
    public void GetHashCode_WithSameValues_ReturnsSameHashCode()
    {
        // Arrange
        var key1 = GroupKey.FromValues(new object[] { "Alice", 25 });
        var key2 = GroupKey.FromValues(new object[] { "Alice", 25 });

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
        var key1 = GroupKey.FromValues(new object[] { "Alice", 25 });
        var key2 = GroupKey.FromValues(new object[] { "Bob", 30 });

        // Act
        var hash1 = key1.GetHashCode();
        var hash2 = key2.GetHashCode();

        // Assert
        Assert.That(hash1, Is.Not.EqualTo(hash2));
    }
}

[TestFixture]
public class TypedGroupKeyTests
{
    [Test]
    public void CreateGroupsInternal_MixedTypeNullKeys_GroupsCorrectly()
    {
        var columns = new Dictionary<string, IColumn>
        {
            ["Name"] = NivaraColumn<string>.Create(new[] { "A", "A", "B", "B", "A", "C" }),
            ["Score"] = NivaraColumn.CreateFromNullable(new int?[] { 1, null, 2, 2, 1, null }),
            ["Weight"] = NivaraColumn<double>.Create(new[] { 1.0, 2.0, 3.0, 3.0, 1.0, 2.0 })
        };

        var grouped = GroupByOperation.CreateGroupsInternal(columns, new[] { "Name", "Score", "Weight" }, offset: 0);

        Assert.That(grouped.GroupCount, Is.EqualTo(4));
        Assert.That(grouped.GetGroupIndices(GroupKey.FromValues(new object?[] { "A", 1, 1.0 })), Is.EquivalentTo(new[] { 0, 4 }));
        Assert.That(grouped.GetGroupIndices(GroupKey.FromValues(new object?[] { "A", null, 2.0 })), Is.EquivalentTo(new[] { 1 }));
        Assert.That(grouped.GetGroupIndices(GroupKey.FromValues(new object?[] { "B", 2, 3.0 })), Is.EquivalentTo(new[] { 2, 3 }));
        Assert.That(grouped.GetGroupIndices(GroupKey.FromValues(new object?[] { "C", null, 2.0 })), Is.EquivalentTo(new[] { 5 }));
    }

    [Test]
    public void CreateGroupsInternal_Offset_AppliesToGroupIndices()
    {
        var columns = new Dictionary<string, IColumn>
        {
            ["K"] = NivaraColumn<string>.Create(new[] { "X", "Y", "X" })
        };

        var grouped = GroupByOperation.CreateGroupsInternal(columns, new[] { "K" }, offset: 10);

        Assert.That(grouped.GetGroupIndices(GroupKey.FromValues(new object[] { "X" })), Is.EquivalentTo(new[] { 10, 12 }));
        Assert.That(grouped.GetGroupIndices(GroupKey.FromValues(new object[] { "Y" })), Is.EquivalentTo(new[] { 11 }));
    }

    [Test]
    public void TypedKey_FromDifferentColumnInstances_AreEqual()
    {
        var columnA = NivaraColumn<int>.Create(new[] { 42, 7 });
        var columnB = NivaraColumn<int>.Create(new[] { 42, 8 });
        var readersA = new IGroupKeyReader[] { GroupKeyReaderFactory.Create(columnA) };
        var readersB = new IGroupKeyReader[] { GroupKeyReaderFactory.Create(columnB) };

        var keyA = new GroupKey(readersA, 0);
        var keyB = new GroupKey(readersB, 0);

        Assert.That(keyA.Equals(keyB), Is.True);
        Assert.That(keyA.GetHashCode(), Is.EqualTo(keyB.GetHashCode()));
    }

    [Test]
    public void TypedKey_NullVsNonNullValue_AreNotEqual()
    {
        var columnA = NivaraColumn.CreateFromNullable(new int?[] { null, 5 });
        var columnB = NivaraColumn.CreateFromNullable(new int?[] { 5, 5 });
        var readersA = new IGroupKeyReader[] { GroupKeyReaderFactory.Create(columnA) };
        var readersB = new IGroupKeyReader[] { GroupKeyReaderFactory.Create(columnB) };

        var keyA = new GroupKey(readersA, 0);
        var keyB = new GroupKey(readersB, 0);

        Assert.That(keyA.Equals(keyB), Is.False);
    }

    [Test]
    public void TypedKey_HashCollisionDifferentRows_ResolvesCorrectly()
    {
        var columnA = NivaraColumn<int>.Create(new[] { 0, 1 });
        var columnB = NivaraColumn<int>.Create(new[] { 31, 0 });

        var grouped = GroupByOperation.CreateGroupsInternal(
            new Dictionary<string, IColumn> { ["A"] = columnA, ["B"] = columnB }, new[] { "A", "B" });

        var readers = new IGroupKeyReader[]
        {
            GroupKeyReaderFactory.Create(columnA),
            GroupKeyReaderFactory.Create(columnB)
        };
        Assert.That(TypedGroupHash.ComputeRowHash(readers, 0), Is.EqualTo(TypedGroupHash.ComputeRowHash(readers, 1)));
        Assert.That(grouped.GroupCount, Is.EqualTo(2));
        Assert.That(grouped.GetGroupIndices(GroupKey.FromValues(new object[] { 0, 31 })), Is.EquivalentTo(new[] { 0 }));
        Assert.That(grouped.GetGroupIndices(GroupKey.FromValues(new object[] { 1, 0 })), Is.EquivalentTo(new[] { 1 }));
    }

    [Test]
    public void TypedKey_Values_BoxesLazily()
    {
        var column = NivaraColumn<string>.Create(new[] { "Alpha" });
        var readers = new IGroupKeyReader[] { GroupKeyReaderFactory.Create(column) };

        var key = new GroupKey(readers, 0);

        Assert.That(key.KeyCount, Is.EqualTo(1));
        Assert.That(key.GetValue(0), Is.EqualTo("Alpha"));
        Assert.That(key.Values, Is.EquivalentTo(new object?[] { "Alpha" }));
        Assert.That(key.ToString(), Is.EqualTo("(Alpha)"));
    }
}
