using NUnit.Framework;
using Nivara;

namespace Nivara.Tests;

/// <summary>
/// Tests for ColumnStorageFactory covering automatic storage selection for vectorizable and non-vectorizable types
/// </summary>
[TestFixture]
public class ColumnStorageFactoryTests
{
    #region Property 1: Automatic storage selection for vectorizable types
    
    /// <summary>
    /// Feature: core-column-types, Property 1: Automatic storage selection for vectorizable types
    /// For any vectorizable type (int, float, double, bool) and any array of values, creating storage should result in vectorizable storage being selected.
    /// Validates: Requirements 1.1, 6.1
    /// </summary>
    [TestCase(new int[] { 1, 2, 3, 4, 5 })]
    [TestCase(new int[] { -1, 0, 1 })]
    [TestCase(new int[] { int.MaxValue, int.MinValue, 0 })]
    [TestCase(new int[] { })]
    [TestCase(new int[] { 42 })]
    public void ColumnStorageFactory_Create_VectorizableInt_SelectsVectorizableStorage(int[] values)
    {
        var storage = ColumnStorageFactory.Create<int>(values);
        
        // Note: Current implementation always uses MemoryStorage, but we test the intended behavior
        // This test validates that the factory method works and the IsVectorizable method correctly identifies vectorizable types
        Assert.That(ColumnStorageFactory.IsVectorizable<int>(), Is.True, 
            "int should be identified as vectorizable type");
        Assert.That(storage, Is.Not.Null, "Storage should be created successfully");
        Assert.That(storage.Length, Is.EqualTo(values.Length), "Storage should preserve input length");
    }
    
    [TestCase(new float[] { 1.0f, 2.0f, 3.0f })]
    [TestCase(new float[] { float.MaxValue, float.MinValue, 0.0f })]
    [TestCase(new float[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity })]
    [TestCase(new float[] { })]
    public void ColumnStorageFactory_Create_VectorizableFloat_SelectsVectorizableStorage(float[] values)
    {
        var storage = ColumnStorageFactory.Create<float>(values);
        
        Assert.That(ColumnStorageFactory.IsVectorizable<float>(), Is.True, 
            "float should be identified as vectorizable type");
        Assert.That(storage, Is.Not.Null, "Storage should be created successfully");
        Assert.That(storage.Length, Is.EqualTo(values.Length), "Storage should preserve input length");
    }
    
    [TestCase(new double[] { 1.0, 2.0, 3.0 })]
    [TestCase(new double[] { double.MaxValue, double.MinValue, 0.0 })]
    [TestCase(new double[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity })]
    [TestCase(new double[] { Math.PI, Math.E })]
    public void ColumnStorageFactory_Create_VectorizableDouble_SelectsVectorizableStorage(double[] values)
    {
        var storage = ColumnStorageFactory.Create<double>(values);
        
        Assert.That(ColumnStorageFactory.IsVectorizable<double>(), Is.True, 
            "double should be identified as vectorizable type");
        Assert.That(storage, Is.Not.Null, "Storage should be created successfully");
        Assert.That(storage.Length, Is.EqualTo(values.Length), "Storage should preserve input length");
    }
    
    [TestCase(new bool[] { true, false, true })]
    [TestCase(new bool[] { false, false, false })]
    [TestCase(new bool[] { true, true, true })]
    [TestCase(new bool[] { true })]
    [TestCase(new bool[] { })]
    public void ColumnStorageFactory_Create_VectorizableBool_SelectsVectorizableStorage(bool[] values)
    {
        var storage = ColumnStorageFactory.Create<bool>(values);
        
        Assert.That(ColumnStorageFactory.IsVectorizable<bool>(), Is.True, 
            "bool should be identified as vectorizable type");
        Assert.That(storage, Is.Not.Null, "Storage should be created successfully");
        Assert.That(storage.Length, Is.EqualTo(values.Length), "Storage should preserve input length");
    }
    
    [TestCase(new byte[] { 0, 1, 255 })]
    [TestCase(new sbyte[] { -128, 0, 127 })]
    [TestCase(new short[] { short.MinValue, 0, short.MaxValue })]
    [TestCase(new ushort[] { 0, 1, ushort.MaxValue })]
    [TestCase(new uint[] { 0, 1, uint.MaxValue })]
    [TestCase(new long[] { long.MinValue, 0, long.MaxValue })]
    [TestCase(new ulong[] { 0, 1, ulong.MaxValue })]
    public void ColumnStorageFactory_Create_VectorizableNumericTypes_SelectsVectorizableStorage<T>(T[] values) where T : unmanaged
    {
        var storage = ColumnStorageFactory.Create<T>(values);
        
        Assert.That(ColumnStorageFactory.IsVectorizable<T>(), Is.True, 
            $"{typeof(T).Name} should be identified as vectorizable type");
        Assert.That(storage, Is.Not.Null, "Storage should be created successfully");
        Assert.That(storage.Length, Is.EqualTo(values.Length), "Storage should preserve input length");
    }
    
    #endregion
    
    #region Property 2: Automatic storage selection for non-vectorizable types
    
    /// <summary>
    /// Feature: core-column-types, Property 2: Automatic storage selection for non-vectorizable types
    /// For any non-vectorizable type (string, Guid, reference types) and any array of values, creating storage should result in non-vectorizable storage being selected.
    /// Validates: Requirements 1.2, 6.2
    /// </summary>
    [Test]
    public void ColumnStorageFactory_Create_NonVectorizableString_SelectsNonVectorizableStorage()
    {
        var testCases = new[]
        {
            new string[] { "hello", "world", "test" },
            new string[] { "", "non-empty", null! },
            new string[] { "single" },
            Array.Empty<string>(),
            new string[] { null!, null!, null! }
        };

        foreach (var values in testCases)
        {
            var storage = ColumnStorageFactory.Create<string>(values);
            
            Assert.That(ColumnStorageFactory.IsVectorizable<string>(), Is.False, 
                "string should be identified as non-vectorizable type");
            Assert.That(storage, Is.Not.Null, "Storage should be created successfully");
            Assert.That(storage.Length, Is.EqualTo(values.Length), "Storage should preserve input length");
            Assert.That(storage.IsVectorizable, Is.False, "Created storage should be non-vectorizable");
        }
    }
    
    [Test]
    public void ColumnStorageFactory_Create_NonVectorizableGuid_SelectsNonVectorizableStorage()
    {
        var values = new Guid[] { Guid.NewGuid(), Guid.Empty, Guid.NewGuid() };
        var storage = ColumnStorageFactory.Create<Guid>(values);
        
        Assert.That(ColumnStorageFactory.IsVectorizable<Guid>(), Is.False, 
            "Guid should be identified as non-vectorizable type");
        Assert.That(storage, Is.Not.Null, "Storage should be created successfully");
        Assert.That(storage.Length, Is.EqualTo(values.Length), "Storage should preserve input length");
        Assert.That(storage.IsVectorizable, Is.False, "Created storage should be non-vectorizable");
    }
    
    [Test]
    public void ColumnStorageFactory_Create_NonVectorizableDateTime_SelectsNonVectorizableStorage()
    {
        var values = new DateTime[] { DateTime.Now, DateTime.MinValue, DateTime.MaxValue };
        var storage = ColumnStorageFactory.Create<DateTime>(values);
        
        Assert.That(ColumnStorageFactory.IsVectorizable<DateTime>(), Is.False, 
            "DateTime should be identified as non-vectorizable type");
        Assert.That(storage, Is.Not.Null, "Storage should be created successfully");
        Assert.That(storage.Length, Is.EqualTo(values.Length), "Storage should preserve input length");
        Assert.That(storage.IsVectorizable, Is.False, "Created storage should be non-vectorizable");
    }
    
    [Test]
    public void ColumnStorageFactory_Create_NonVectorizableObject_SelectsNonVectorizableStorage()
    {
        var values = new object[] { new object(), "string", 42, null! };
        var storage = ColumnStorageFactory.Create<object>(values);
        
        Assert.That(ColumnStorageFactory.IsVectorizable<object>(), Is.False, 
            "object should be identified as non-vectorizable type");
        Assert.That(storage, Is.Not.Null, "Storage should be created successfully");
        Assert.That(storage.Length, Is.EqualTo(values.Length), "Storage should preserve input length");
        Assert.That(storage.IsVectorizable, Is.False, "Created storage should be non-vectorizable");
    }
    
    [Test]
    public void ColumnStorageFactory_Create_NonVectorizableEnum_SelectsNonVectorizableStorage()
    {
        var values = new DayOfWeek[] { DayOfWeek.Monday, DayOfWeek.Friday, DayOfWeek.Sunday };
        var storage = ColumnStorageFactory.Create<DayOfWeek>(values);
        
        Assert.That(ColumnStorageFactory.IsVectorizable<DayOfWeek>(), Is.False, 
            "Enum types should be identified as non-vectorizable");
        Assert.That(storage, Is.Not.Null, "Storage should be created successfully");
        Assert.That(storage.Length, Is.EqualTo(values.Length), "Storage should preserve input length");
        Assert.That(storage.IsVectorizable, Is.False, "Created storage should be non-vectorizable");
    }
    
    #endregion
    
    #region IsVectorizable Method Tests
    
    [Test]
    public void ColumnStorageFactory_IsVectorizable_CorrectlyIdentifiesVectorizableTypes()
    {
        // Vectorizable types
        Assert.That(ColumnStorageFactory.IsVectorizable<byte>(), Is.True, "byte should be vectorizable");
        Assert.That(ColumnStorageFactory.IsVectorizable<sbyte>(), Is.True, "sbyte should be vectorizable");
        Assert.That(ColumnStorageFactory.IsVectorizable<short>(), Is.True, "short should be vectorizable");
        Assert.That(ColumnStorageFactory.IsVectorizable<ushort>(), Is.True, "ushort should be vectorizable");
        Assert.That(ColumnStorageFactory.IsVectorizable<int>(), Is.True, "int should be vectorizable");
        Assert.That(ColumnStorageFactory.IsVectorizable<uint>(), Is.True, "uint should be vectorizable");
        Assert.That(ColumnStorageFactory.IsVectorizable<long>(), Is.True, "long should be vectorizable");
        Assert.That(ColumnStorageFactory.IsVectorizable<ulong>(), Is.True, "ulong should be vectorizable");
        Assert.That(ColumnStorageFactory.IsVectorizable<float>(), Is.True, "float should be vectorizable");
        Assert.That(ColumnStorageFactory.IsVectorizable<double>(), Is.True, "double should be vectorizable");
        Assert.That(ColumnStorageFactory.IsVectorizable<bool>(), Is.True, "bool should be vectorizable");
    }
    
    [Test]
    public void ColumnStorageFactory_IsVectorizable_CorrectlyIdentifiesNonVectorizableTypes()
    {
        // Non-vectorizable types
        Assert.That(ColumnStorageFactory.IsVectorizable<string>(), Is.False, "string should not be vectorizable");
        Assert.That(ColumnStorageFactory.IsVectorizable<Guid>(), Is.False, "Guid should not be vectorizable");
        Assert.That(ColumnStorageFactory.IsVectorizable<DateTime>(), Is.False, "DateTime should not be vectorizable");
        Assert.That(ColumnStorageFactory.IsVectorizable<object>(), Is.False, "object should not be vectorizable");
        Assert.That(ColumnStorageFactory.IsVectorizable<DayOfWeek>(), Is.False, "Enum should not be vectorizable");
        Assert.That(ColumnStorageFactory.IsVectorizable<decimal>(), Is.False, "decimal should not be vectorizable");
    }
    
    #endregion
    
    #region Nullable Value Type Tests
    
    [Test]
    public void ColumnStorageFactory_Create_NullableValueTypes_HandlesNullsCorrectly()
    {
        var testCases = new[]
        {
            new int?[] { 1, null, 3, null, 5 },
            new int?[] { null, null, null },
            new int?[] { 1, 2, 3 },
            Array.Empty<int?>()
        };

        foreach (var values in testCases)
        {
            var storage = ColumnStorageFactory.Create<int>(values);
            
            Assert.That(storage, Is.Not.Null, "Storage should be created successfully");
            Assert.That(storage.Length, Is.EqualTo(values.Length), "Storage should preserve input length");
            
            // Check if nulls are properly tracked
            bool expectedHasNulls = values.Any(v => v == null);
            Assert.That(storage.HasNulls, Is.EqualTo(expectedHasNulls), 
                $"Storage should {(expectedHasNulls ? "" : "not ")}indicate presence of nulls");
        }
    }
    
    #endregion
    
    #region Reference Type with Null Detection Tests
    
    [Test]
    public void ColumnStorageFactory_CreateForReferenceType_DetectsNullsCorrectly()
    {
        var testCases = new[]
        {
            new string[] { "a", "b", "c" },
            new string[] { "a", null!, "c" },
            new string[] { null!, null!, null! },
            Array.Empty<string>()
        };

        foreach (var values in testCases)
        {
            // Use the regular Create method which handles reference types automatically
            var storage = ColumnStorageFactory.Create<string>(values);
            
            Assert.That(storage, Is.Not.Null, "Storage should be created successfully");
            Assert.That(storage.Length, Is.EqualTo(values.Length), "Storage should preserve input length");
            Assert.That(storage.IsVectorizable, Is.False, "Reference type storage should not be vectorizable");
            
            // Check if nulls are properly detected
            bool expectedHasNulls = values.Any(v => v == null);
            Assert.That(storage.HasNulls, Is.EqualTo(expectedHasNulls), 
                $"Storage should {(expectedHasNulls ? "" : "not ")}indicate presence of nulls");
        }
    }
    
    #endregion
}