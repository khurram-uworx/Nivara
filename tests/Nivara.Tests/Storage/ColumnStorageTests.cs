using Nivara.Storage;
using NUnit.Framework;

namespace Nivara.Tests.Storage;

/// <summary>
/// Tests for ColumnStorage implementation covering edge cases for string columns, empty arrays, single-element arrays, and null handling for reference types
/// </summary>
[TestFixture]
public class ColumnStorageTests
{
    #region String Column Tests - Requirements 1.2, 1.3

    [Test]
    public void ColumnStorage_StringColumns_StoresAndRetrievesCorrectly()
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
            var storage = new ColumnStorage<string>(values, detectNulls: true);

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
    public void ColumnStorage_StringColumns_HandlesNullsCorrectly()
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
            var storage = new ColumnStorage<string>(testCase.Values, detectNulls: true);

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
    public void ColumnStorage_EmptyStringArray_HandlesCorrectly()
    {
        var storage = new ColumnStorage<string>(Array.Empty<string>(), detectNulls: true);

        Assert.That(storage.Length, Is.EqualTo(0), "Empty storage should have zero length");
        Assert.That(storage.IsVectorizable, Is.False, "String storage should not be vectorizable");
        Assert.That(storage.HasNulls, Is.False, "Empty storage should not have nulls");
        Assert.That(storage.NullMask.Length, Is.EqualTo(0), "Empty storage should have empty null mask");
    }

    [Test]
    public void ColumnStorage_EmptyIntArray_HandlesCorrectly()
    {
        var storage = new ColumnStorage<int>(Array.Empty<int>(), detectNulls: false);

        Assert.That(storage.Length, Is.EqualTo(0), "Empty storage should have zero length");
        Assert.That(storage.IsVectorizable, Is.True, "ColumnStorage should reflect element type vectorizability");
        Assert.That(storage.HasNulls, Is.False, "Empty storage should not have nulls");
        Assert.That(storage.NullMask.Length, Is.EqualTo(0), "Empty storage should have empty null mask");
    }

    [Test]
    public void ColumnStorage_EmptyObjectArray_HandlesCorrectly()
    {
        var storage = new ColumnStorage<object>(Array.Empty<object>(), detectNulls: true);

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
    public void ColumnStorage_SingleStringElement_HandlesCorrectly(string value)
    {
        var storage = new ColumnStorage<string>(new[] { value }, detectNulls: true);

        Assert.That(storage.Length, Is.EqualTo(1), "Single element storage should have length 1");
        Assert.That(storage.IsVectorizable, Is.False, "String storage should not be vectorizable");
        Assert.That(storage[0], Is.EqualTo(value), "Single element should be retrievable");
        Assert.That(storage.HasNulls, Is.False, "Non-null single element should not indicate nulls");
    }

    [Test]
    public void ColumnStorage_SingleNullStringElement_HandlesCorrectly()
    {
        var storage = new ColumnStorage<string>(new string[] { null! }, detectNulls: true);

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
    public void ColumnStorage_SingleIntElement_HandlesCorrectly(int value)
    {
        var storage = new ColumnStorage<int>(new[] { value }, detectNulls: false);

        Assert.That(storage.Length, Is.EqualTo(1), "Single element storage should have length 1");
        Assert.That(storage.IsVectorizable, Is.True, "ColumnStorage should reflect element type vectorizability");
        Assert.That(storage[0], Is.EqualTo(value), "Single element should be retrievable");
        Assert.That(storage.HasNulls, Is.False, "Value type storage without null detection should not indicate nulls");
    }

    [Test]
    public void ColumnStorage_SingleObjectElement_HandlesCorrectly()
    {
        var obj = new object();
        var storage = new ColumnStorage<object>(new[] { obj }, detectNulls: true);

        Assert.That(storage.Length, Is.EqualTo(1), "Single element storage should have length 1");
        Assert.That(storage.IsVectorizable, Is.False, "Object storage should not be vectorizable");
        Assert.That(storage[0], Is.SameAs(obj), "Single element should be retrievable");
        Assert.That(storage.HasNulls, Is.False, "Non-null single element should not indicate nulls");
    }

    #endregion

    #region Reference Type Null Handling Tests - Requirements 1.2, 1.3

    [Test]
    public void ColumnStorage_ObjectArray_WithNulls_HandlesCorrectly()
    {
        var obj1 = new object();
        var obj2 = new object();
        var values = new object[] { obj1, null!, obj2, null! };
        var storage = new ColumnStorage<object>(values, detectNulls: true);

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
    public void ColumnStorage_ObjectArray_WithoutNulls_HandlesCorrectly()
    {
        var obj1 = new object();
        var obj2 = new object();
        var obj3 = new object();
        var values = new object[] { obj1, obj2, obj3 };
        var storage = new ColumnStorage<object>(values, detectNulls: true);

        Assert.That(storage.Length, Is.EqualTo(3), "Storage should have correct length");
        Assert.That(storage.HasNulls, Is.False, "Storage should not indicate presence of nulls");
        Assert.That(storage.NullMask.Length, Is.EqualTo(0), "Null mask should be empty");

        Assert.That(storage[0], Is.SameAs(obj1), "First element should be correct object");
        Assert.That(storage[1], Is.SameAs(obj2), "Second element should be correct object");
        Assert.That(storage[2], Is.SameAs(obj3), "Third element should be correct object");
    }

    [Test]
    public void ColumnStorage_ReferenceType_WithoutNullDetection_DoesNotTrackNulls()
    {
        var values = new string[] { "hello", null!, "world" };
        var storage = new ColumnStorage<string>(values, detectNulls: false);

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
    public void ColumnStorage_IndexerAccess_ThrowsForInvalidIndex(string[] values, int invalidIndex)
    {
        var storage = new ColumnStorage<string>(values, detectNulls: true);

        Assert.Throws<IndexOutOfRangeException>(() => _ = storage[invalidIndex],
            $"Accessing index {invalidIndex} should throw IndexOutOfRangeException");
    }

    [TestCase(new int[] { 1, 2, 3, 4, 5 }, 0, 1)]
    [TestCase(new int[] { 1, 2, 3, 4, 5 }, 2, 3)]
    [TestCase(new int[] { 1, 2, 3, 4, 5 }, 4, 5)]
    public void ColumnStorage_IndexerAccess_ReturnsCorrectValues(int[] values, int index, int expectedValue)
    {
        var storage = new ColumnStorage<int>(values, detectNulls: false);

        Assert.That(storage[index], Is.EqualTo(expectedValue),
            $"Value at index {index} should be {expectedValue}");
    }

    #endregion

    #region Slicing Tests

    [TestCase(new string[] { "a", "b", "c", "d", "e" }, 1, 3)]
    [TestCase(new string[] { "a", "b", "c", "d", "e" }, 0, 2)]
    [TestCase(new string[] { "a", "b", "c", "d", "e" }, 2, 3)]
    [TestCase(new string[] { "a", "b", "c", "d", "e" }, 0, 5)]
    public void ColumnStorage_Slice_ReturnsCorrectSubset(string[] values, int start, int length)
    {
        var storage = new ColumnStorage<string>(values, detectNulls: true);

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
    public void ColumnStorage_Slice_PreservesNullMask()
    {
        var testCases = new[]
        {
            new { Values = new string[] { "a", null!, "c" }, Start = 0, Length = 3, Description = "full slice with nulls" },
            new { Values = new string[] { "a", null!, "c" }, Start = 1, Length = 2, Description = "partial slice with nulls" }
        };

        foreach (var testCase in testCases)
        {
            var storage = new ColumnStorage<string>(testCase.Values, detectNulls: true);

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

    [Test]
    public void ColumnStorage_ExplicitEmptyNullMask_TreatsAsNoNullMask()
    {
        var storage = new ColumnStorage<int>(
            new ReadOnlyMemory<int>(new[] { 10, 20, 30 }),
            ReadOnlyMemory<bool>.Empty);

        Assert.That(storage.HasNulls, Is.False);
        Assert.That(storage.NullMask.Length, Is.EqualTo(0));

        var sliced = storage.Slice(1, 2);

        Assert.That(sliced.HasNulls, Is.False);
        Assert.That(sliced.NullMask.Length, Is.EqualTo(0));
        Assert.That(sliced[0], Is.EqualTo(20));
        Assert.That(sliced[1], Is.EqualTo(30));
    }

    [TestCase(new string[] { "a", "b", "c" }, -1, 1)]
    [TestCase(new string[] { "a", "b", "c" }, 0, -1)]
    [TestCase(new string[] { "a", "b", "c" }, 2, 3)]
    [TestCase(new string[] { "a", "b", "c" }, 4, 1)]
    public void ColumnStorage_Slice_ThrowsForInvalidParameters(string[] values, int start, int length)
    {
        var storage = new ColumnStorage<string>(values, detectNulls: true);

        Assert.Throws<ArgumentOutOfRangeException>(() => storage.Slice(start, length),
            $"Slice with start={start}, length={length} should throw ArgumentOutOfRangeException");
    }

    #endregion

    #region Disposal Tests

    [Test]
    public void ColumnStorage_Dispose_PreventsAccess()
    {
        var storage = new ColumnStorage<string>(new[] { "a", "b", "c" }, detectNulls: true);

        storage.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = storage[0],
            "Accessing disposed storage should throw ObjectDisposedException");
        Assert.Throws<ObjectDisposedException>(() => _ = storage.NullMask,
            "Accessing null mask of disposed storage should throw ObjectDisposedException");
        Assert.Throws<ObjectDisposedException>(() => storage.Slice(0, 1),
            "Slicing disposed storage should throw ObjectDisposedException");
    }

    #endregion

    #region Span access & slice view semantics

    [Test]
    public void ColumnStorage_Slice_ReturnsSharedBufferView()
    {
        var sourceArray = new[] { 10, 20, 30, 40 };
        var storage = new ColumnStorage<int>(new ReadOnlyMemory<int>(sourceArray));

        var sliced = storage.Slice(1, 2);
        Assert.That(sliced[0], Is.EqualTo(20));

        sourceArray[1] = 200;

        Assert.That(sliced[0], Is.EqualTo(200),
            "ColumnStorage.Slice should share the underlying buffer (zero-copy view)");
    }

    [Test]
    public void ColumnStorage_Slice_WithNulls_ReturnsSharedNullMaskView()
    {
        var dataArray = new[] { 1, 2, 3, 4 };
        var maskArray = new bool[] { false, true, false, false };
        var storage = new ColumnStorage<int>(new ReadOnlyMemory<int>(dataArray), new ReadOnlyMemory<bool>(maskArray));

        var sliced = storage.Slice(1, 2);
        Assert.That(sliced.HasNulls, Is.True);
        Assert.That(sliced.NullMask[0], Is.True);

        maskArray[1] = false;

        Assert.That(sliced.NullMask[0], Is.False,
            "ColumnStorage null-mask slice should share the underlying buffer");
    }

    [Test]
    public void ColumnStorage_AsTensor_ReturnsZeroCopyView()
    {
        var sourceArray = new[] { 1, 2, 3, 4, 5 };
        var storage = new ColumnStorage<int>(sourceArray);

        var tensor = storage.AsTensor();
        Assert.That(tensor.TryGetSpan(new nint[] { 0 }, (int)tensor.FlattenedLength, out Span<int> tensorSpan), Is.True);

        ((IColumnStorage<int>)storage).TryGetSpan(out var storageSpan);

        Assert.That(tensorSpan == storageSpan, Is.True,
            "AsTensor() must share the storage's backing array (zero-copy view)");
    }

    [Test]
    public void ColumnStorage_AsTensor_IsLazyCached()
    {
        var storage = new ColumnStorage<double>(new[] { 1.0, 2.0, 3.0 });

        var first = storage.AsTensor();
        var second = storage.AsTensor();

        Assert.That(ReferenceEquals(first, second), Is.True,
            "AsTensor() should return the cached view on repeated access");
    }

    [Test]
    public void ColumnStorage_AsTensor_ThrowsForReferenceContainingTypes()
    {
        var storage = new ColumnStorage<string>(new[] { "a", "b" }, detectNulls: true);

        var ex = Assert.Throws<InvalidOperationException>(() => storage.AsTensor());
        Assert.That(ex!.Message, Does.Contain("unmanaged"),
            "AsTensor() guard should reject reference-containing element types");
    }

    [Test]
    public void ColumnStorage_UnifiedConstruction_ValueTypeWithoutMaskAndRefTypeWithMask()
    {
        var floatNoMask = new ColumnStorage<float>(new[] { 1.0f, 2.0f, 3.0f });
        Assert.That(floatNoMask.HasNulls, Is.False);
        Assert.That(floatNoMask.Length, Is.EqualTo(3));
        Assert.That(floatNoMask.IsVectorizable, Is.True);

        var stringWithMask = new ColumnStorage<string>(new string[] { "a", null!, "c" }, detectNulls: true);
        Assert.That(stringWithMask.HasNulls, Is.True);
        Assert.That(stringWithMask.NullMask.Length, Is.EqualTo(3));
        Assert.That(stringWithMask.NullMask[1], Is.True);
        Assert.That(stringWithMask.IsVectorizable, Is.False);

        var half = new ColumnStorage<Half>(new[] { (Half)1.0, (Half)2.0 });
        Assert.That(half.IsVectorizable, Is.False, "Half is unmanaged but not a confirmed vectorizable type");
        Assert.DoesNotThrow(() => half.AsTensor(), "Half is unmanaged so the AsTensor guard must pass");
    }

    [Test]
    public void ColumnStorage_IsVectorizable_ReflectsElementType_NotBackend()
    {
        // Vectorizable element types must report true even in Memory-backed storage,
        // so kernel dispatch depends on the element type (issue #102).
        Assert.That(new ColumnStorage<int>(new[] { 1, 2, 3 }).IsVectorizable, Is.True);
        Assert.That(new ColumnStorage<double>(new[] { 1.0, 2.0 }).IsVectorizable, Is.True);
        Assert.That(new ColumnStorage<bool>(new[] { true, false }).IsVectorizable, Is.True);
        Assert.That(new ColumnStorage<long>(new[] { 1L, 2L }).IsVectorizable, Is.True);

        // Non-vectorizable element types stay false.
        Assert.That(new ColumnStorage<string>(new[] { "a" }, detectNulls: true).IsVectorizable, Is.False);
        Assert.That(new ColumnStorage<Guid>(new[] { Guid.Empty }).IsVectorizable, Is.False);
        Assert.That(new ColumnStorage<decimal>(new[] { 1.0m }).IsVectorizable, Is.False);
    }

    #endregion
}
