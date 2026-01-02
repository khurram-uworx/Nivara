using Nivara.Storage;
using NUnit.Framework;

namespace Nivara.Tests.Storage;

/// <summary>
/// Tests for MemoryStorage implementation covering edge cases for string columns, empty arrays, single-element arrays, and null handling for reference types
/// </summary>
[TestFixture]
public class MemoryStorageTests
{
    #region String Column Tests - Requirements 1.2, 1.3

    [Test]
    public void MemoryStorage_StringColumns_StoresAndRetrievesCorrectly()
    {
        var testCases = new[]
        {
            new string[] { "hello", "world", "test" },
            new string[] { "single" },
            new string[] { "", "non-empty", "another" },
            new string[] { "a", "bb", "ccc", "dddd" }
        };

        foreach (var values in testCases)
        {
            var storage = new MemoryStorage<string>(values, detectNulls: true);

            Assert.That(storage.Length, Is.EqualTo(values.Length), "Storage should preserve input length");
            Assert.That(storage.IsVectorizable, Is.False, "String storage should not be vectorizable");

            for (int i = 0; i < values.Length; i++)
            {
                Assert.That(storage[i], Is.EqualTo(values[i]),
                    $"Value at index {i} should be '{values[i]}'");
            }
        }
    }

    [Test]
    public void MemoryStorage_StringColumns_HandlesNullsCorrectly()
    {
        var testCases = new[]
        {
            new { Values = new string[] { "hello", null!, "world" }, Description = "mixed nulls" },
            new { Values = new string[] { null!, "test", null! }, Description = "nulls at ends" },
            new { Values = new string[] { null!, null!, null! }, Description = "all nulls" },
            new { Values = new string[] { "only-non-null" }, Description = "single non-null" },
            new { Values = new string[] { null! }, Description = "single null" }
        };

        foreach (var testCase in testCases)
        {
            var storage = new MemoryStorage<string>(testCase.Values, detectNulls: true);

            bool expectedHasNulls = testCase.Values.Any(v => v == null);
            Assert.That(storage.HasNulls, Is.EqualTo(expectedHasNulls),
                $"Storage should {(expectedHasNulls ? "" : "not ")}indicate presence of nulls for {testCase.Description}");

            if (expectedHasNulls)
            {
                Assert.That(storage.NullMask.Length, Is.EqualTo(testCase.Values.Length),
                    $"Null mask length should match data length for {testCase.Description}");

                for (int i = 0; i < testCase.Values.Length; i++)
                {
                    bool expectedIsNull = testCase.Values[i] == null;
                    Assert.That(storage.NullMask[i], Is.EqualTo(expectedIsNull),
                        $"Null mask at index {i} should be {expectedIsNull} for {testCase.Description}");
                }
            }
            else
            {
                Assert.That(storage.NullMask.Length, Is.EqualTo(0),
                    $"Null mask should be empty when no nulls are present for {testCase.Description}");
            }
        }
    }

    #endregion

    #region Empty Array Tests - Requirements 1.2, 1.3

    [Test]
    public void MemoryStorage_EmptyStringArray_HandlesCorrectly()
    {
        var storage = new MemoryStorage<string>(Array.Empty<string>(), detectNulls: true);

        Assert.That(storage.Length, Is.EqualTo(0), "Empty storage should have zero length");
        Assert.That(storage.IsVectorizable, Is.False, "String storage should not be vectorizable");
        Assert.That(storage.HasNulls, Is.False, "Empty storage should not have nulls");
        Assert.That(storage.NullMask.Length, Is.EqualTo(0), "Empty storage should have empty null mask");
    }

    [Test]
    public void MemoryStorage_EmptyIntArray_HandlesCorrectly()
    {
        var storage = new MemoryStorage<int>(Array.Empty<int>(), detectNulls: false);

        Assert.That(storage.Length, Is.EqualTo(0), "Empty storage should have zero length");
        Assert.That(storage.IsVectorizable, Is.False, "MemoryStorage should not be vectorizable");
        Assert.That(storage.HasNulls, Is.False, "Empty storage should not have nulls");
        Assert.That(storage.NullMask.Length, Is.EqualTo(0), "Empty storage should have empty null mask");
    }

    [Test]
    public void MemoryStorage_EmptyObjectArray_HandlesCorrectly()
    {
        var storage = new MemoryStorage<object>(Array.Empty<object>(), detectNulls: true);

        Assert.That(storage.Length, Is.EqualTo(0), "Empty storage should have zero length");
        Assert.That(storage.IsVectorizable, Is.False, "Object storage should not be vectorizable");
        Assert.That(storage.HasNulls, Is.False, "Empty storage should not have nulls");
        Assert.That(storage.NullMask.Length, Is.EqualTo(0), "Empty storage should have empty null mask");
    }

    #endregion

    #region Single Element Array Tests - Requirements 1.2, 1.3

    [TestCase("hello")]
    [TestCase("")]
    [TestCase("single-element")]
    public void MemoryStorage_SingleStringElement_HandlesCorrectly(string value)
    {
        var storage = new MemoryStorage<string>(new[] { value }, detectNulls: true);

        Assert.That(storage.Length, Is.EqualTo(1), "Single element storage should have length 1");
        Assert.That(storage.IsVectorizable, Is.False, "String storage should not be vectorizable");
        Assert.That(storage[0], Is.EqualTo(value), "Single element should be retrievable");
        Assert.That(storage.HasNulls, Is.False, "Non-null single element should not indicate nulls");
    }

    [Test]
    public void MemoryStorage_SingleNullStringElement_HandlesCorrectly()
    {
        var storage = new MemoryStorage<string>(new string[] { null! }, detectNulls: true);

        Assert.That(storage.Length, Is.EqualTo(1), "Single element storage should have length 1");
        Assert.That(storage.IsVectorizable, Is.False, "String storage should not be vectorizable");
        Assert.That(storage[0], Is.Null, "Single null element should be retrievable as null");
        Assert.That(storage.HasNulls, Is.True, "Null single element should indicate nulls");
        Assert.That(storage.NullMask[0], Is.True, "Null mask should indicate null at position 0");
    }

    [TestCase(42)]
    [TestCase(0)]
    [TestCase(-1)]
    [TestCase(int.MaxValue)]
    [TestCase(int.MinValue)]
    public void MemoryStorage_SingleIntElement_HandlesCorrectly(int value)
    {
        var storage = new MemoryStorage<int>(new[] { value }, detectNulls: false);

        Assert.That(storage.Length, Is.EqualTo(1), "Single element storage should have length 1");
        Assert.That(storage.IsVectorizable, Is.False, "MemoryStorage should not be vectorizable");
        Assert.That(storage[0], Is.EqualTo(value), "Single element should be retrievable");
        Assert.That(storage.HasNulls, Is.False, "Value type storage without null detection should not indicate nulls");
    }

    [Test]
    public void MemoryStorage_SingleObjectElement_HandlesCorrectly()
    {
        var obj = new object();
        var storage = new MemoryStorage<object>(new[] { obj }, detectNulls: true);

        Assert.That(storage.Length, Is.EqualTo(1), "Single element storage should have length 1");
        Assert.That(storage.IsVectorizable, Is.False, "Object storage should not be vectorizable");
        Assert.That(storage[0], Is.SameAs(obj), "Single element should be retrievable");
        Assert.That(storage.HasNulls, Is.False, "Non-null single element should not indicate nulls");
    }

    #endregion

    #region Reference Type Null Handling Tests - Requirements 1.2, 1.3

    [Test]
    public void MemoryStorage_ObjectArray_WithNulls_HandlesCorrectly()
    {
        var obj1 = new object();
        var obj2 = new object();
        var values = new object[] { obj1, null!, obj2, null! };
        var storage = new MemoryStorage<object>(values, detectNulls: true);

        Assert.That(storage.Length, Is.EqualTo(4), "Storage should have correct length");
        Assert.That(storage.HasNulls, Is.True, "Storage should indicate presence of nulls");
        Assert.That(storage.NullMask.Length, Is.EqualTo(4), "Null mask should have correct length");

        Assert.That(storage[0], Is.SameAs(obj1), "First element should be correct object");
        Assert.That(storage[1], Is.Null, "Second element should be null");
        Assert.That(storage[2], Is.SameAs(obj2), "Third element should be correct object");
        Assert.That(storage[3], Is.Null, "Fourth element should be null");

        Assert.That(storage.NullMask[0], Is.False, "First element should not be marked as null");
        Assert.That(storage.NullMask[1], Is.True, "Second element should be marked as null");
        Assert.That(storage.NullMask[2], Is.False, "Third element should not be marked as null");
        Assert.That(storage.NullMask[3], Is.True, "Fourth element should be marked as null");
    }

    [Test]
    public void MemoryStorage_ObjectArray_WithoutNulls_HandlesCorrectly()
    {
        var obj1 = new object();
        var obj2 = new object();
        var obj3 = new object();
        var values = new object[] { obj1, obj2, obj3 };
        var storage = new MemoryStorage<object>(values, detectNulls: true);

        Assert.That(storage.Length, Is.EqualTo(3), "Storage should have correct length");
        Assert.That(storage.HasNulls, Is.False, "Storage should not indicate presence of nulls");
        Assert.That(storage.NullMask.Length, Is.EqualTo(0), "Null mask should be empty");

        Assert.That(storage[0], Is.SameAs(obj1), "First element should be correct object");
        Assert.That(storage[1], Is.SameAs(obj2), "Second element should be correct object");
        Assert.That(storage[2], Is.SameAs(obj3), "Third element should be correct object");
    }

    [Test]
    public void MemoryStorage_ReferenceType_WithoutNullDetection_DoesNotTrackNulls()
    {
        var values = new string[] { "hello", null!, "world" };
        var storage = new MemoryStorage<string>(values, detectNulls: false);

        Assert.That(storage.Length, Is.EqualTo(3), "Storage should have correct length");
        Assert.That(storage.HasNulls, Is.False, "Storage should not indicate nulls when detection is disabled");
        Assert.That(storage.NullMask.Length, Is.EqualTo(0), "Null mask should be empty when detection is disabled");

        Assert.That(storage[0], Is.EqualTo("hello"), "First element should be correct");
        Assert.That(storage[1], Is.Null, "Second element should be null (but not tracked)");
        Assert.That(storage[2], Is.EqualTo("world"), "Third element should be correct");
    }

    #endregion

    #region Indexer Access and Bounds Checking Tests

    [TestCase(new string[] { "a", "b", "c" }, -1)]
    [TestCase(new string[] { "a", "b", "c" }, 3)]
    [TestCase(new string[] { "a", "b", "c" }, 10)]
    [TestCase(new string[] { }, 0)]
    public void MemoryStorage_IndexerAccess_ThrowsForInvalidIndex(string[] values, int invalidIndex)
    {
        var storage = new MemoryStorage<string>(values, detectNulls: true);

        Assert.Throws<IndexOutOfRangeException>(() => _ = storage[invalidIndex],
            $"Accessing index {invalidIndex} should throw IndexOutOfRangeException");
    }

    [TestCase(new int[] { 1, 2, 3, 4, 5 }, 0, 1)]
    [TestCase(new int[] { 1, 2, 3, 4, 5 }, 2, 3)]
    [TestCase(new int[] { 1, 2, 3, 4, 5 }, 4, 5)]
    public void MemoryStorage_IndexerAccess_ReturnsCorrectValues(int[] values, int index, int expectedValue)
    {
        var storage = new MemoryStorage<int>(values, detectNulls: false);

        Assert.That(storage[index], Is.EqualTo(expectedValue),
            $"Value at index {index} should be {expectedValue}");
    }

    #endregion

    #region Slicing Tests

    [TestCase(new string[] { "a", "b", "c", "d", "e" }, 1, 3)]
    [TestCase(new string[] { "a", "b", "c", "d", "e" }, 0, 2)]
    [TestCase(new string[] { "a", "b", "c", "d", "e" }, 2, 3)]
    [TestCase(new string[] { "a", "b", "c", "d", "e" }, 0, 5)]
    public void MemoryStorage_Slice_ReturnsCorrectSubset(string[] values, int start, int length)
    {
        var storage = new MemoryStorage<string>(values, detectNulls: true);

        var sliced = storage.Slice(start, length);

        Assert.That(sliced.Length, Is.EqualTo(length), "Sliced storage should have correct length");
        Assert.That(sliced.IsVectorizable, Is.False, "Sliced storage should maintain vectorizable property");

        for (int i = 0; i < length; i++)
        {
            Assert.That(sliced[i], Is.EqualTo(values[start + i]),
                $"Sliced value at index {i} should match original value at index {start + i}");
        }
    }

    [Test]
    public void MemoryStorage_Slice_PreservesNullMask()
    {
        var testCases = new[]
        {
            new { Values = new string[] { "a", null!, "c" }, Start = 0, Length = 3, Description = "full slice with nulls" },
            new { Values = new string[] { "a", null!, "c" }, Start = 1, Length = 2, Description = "partial slice with nulls" }
        };

        foreach (var testCase in testCases)
        {
            var storage = new MemoryStorage<string>(testCase.Values, detectNulls: true);

            var sliced = storage.Slice(testCase.Start, testCase.Length);

            bool expectedHasNulls = testCase.Values.Skip(testCase.Start).Take(testCase.Length).Any(v => v == null);
            Assert.That(sliced.HasNulls, Is.EqualTo(expectedHasNulls),
                $"Sliced storage should correctly indicate null presence for {testCase.Description}");

            if (expectedHasNulls)
            {
                for (int i = 0; i < testCase.Length; i++)
                {
                    bool expectedIsNull = testCase.Values[testCase.Start + i] == null;
                    Assert.That(sliced.NullMask[i], Is.EqualTo(expectedIsNull),
                        $"Sliced null mask at index {i} should be {expectedIsNull} for {testCase.Description}");
                }
            }
        }
    }

    [TestCase(new string[] { "a", "b", "c" }, -1, 1)]
    [TestCase(new string[] { "a", "b", "c" }, 0, -1)]
    [TestCase(new string[] { "a", "b", "c" }, 2, 3)]
    [TestCase(new string[] { "a", "b", "c" }, 4, 1)]
    public void MemoryStorage_Slice_ThrowsForInvalidParameters(string[] values, int start, int length)
    {
        var storage = new MemoryStorage<string>(values, detectNulls: true);

        Assert.Throws<ArgumentOutOfRangeException>(() => storage.Slice(start, length),
            $"Slice with start={start}, length={length} should throw ArgumentOutOfRangeException");
    }

    #endregion

    #region Disposal Tests

    [Test]
    public void MemoryStorage_Dispose_PreventsAccess()
    {
        var storage = new MemoryStorage<string>(new[] { "a", "b", "c" }, detectNulls: true);

        storage.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = storage[0],
            "Accessing disposed storage should throw ObjectDisposedException");
        Assert.Throws<ObjectDisposedException>(() => _ = storage.NullMask,
            "Accessing null mask of disposed storage should throw ObjectDisposedException");
        Assert.Throws<ObjectDisposedException>(() => storage.Slice(0, 1),
            "Slicing disposed storage should throw ObjectDisposedException");
    }

    #endregion
}
