using NUnit.Framework;
using Nivara;

namespace Nivara.Tests;

/// <summary>
/// Additional error condition tests for null handling methods.
/// Tests error conditions for Requirements 6.5 that are specific to null handling.
/// </summary>
[TestFixture]
public class NullHandlingErrorConditionTests
{
    #region FillNull Error Conditions

    /// <summary>
    /// Test that FillNull with null fillValue for reference types throws ArgumentNullException
    /// **Validates: Requirements 6.5**
    /// </summary>
    [Test]
    public void FillNull_WithNullFillValueForReferenceTypes_ShouldThrowArgumentNullException()
    {
        var stringColumn = NivaraColumn<string>.Create(new string?[] { "a", null, "c" }!);
        
        var ex = Assert.Throws<ArgumentNullException>(() => stringColumn.FillNull(null!),
            "FillNull with null fillValue for reference types should throw ArgumentNullException");
        
        Assert.That(ex.ParamName, Is.EqualTo("fillValue"),
            "Exception should specify the fillValue parameter");
        Assert.That(ex.Message, Does.Contain("Fill value cannot be null for reference type String"),
            "Error message should explain the issue clearly");
    }

    /// <summary>
    /// Test that FillNull on disposed column throws ObjectDisposedException
    /// **Validates: Requirements 6.5**
    /// </summary>
    [Test]
    public void FillNull_OnDisposedColumn_ShouldThrowObjectDisposedException()
    {
        var column = NivaraColumn<int>.CreateFromNullable(new int?[] { 1, null, 3 });
        column.Dispose();
        
        Assert.Throws<ObjectDisposedException>(() => column.FillNull(42),
            "FillNull on disposed column should throw ObjectDisposedException");
    }

    #endregion

    #region FillNullForward Error Conditions

    /// <summary>
    /// Test that FillNullForward throws InvalidOperationException when first element is null
    /// **Validates: Requirements 6.5**
    /// </summary>
    [Test]
    public void FillNullForward_WithFirstElementNull_ShouldThrowInvalidOperationException()
    {
        var column = NivaraColumn<int>.CreateFromNullable(new int?[] { null, 2, 3 });
        
        var ex = Assert.Throws<InvalidOperationException>(() => column.FillNullForward(),
            "FillNullForward with first element null should throw InvalidOperationException");
        
        Assert.That(ex.Message, Does.Contain("no preceding non-null value"),
            "Error message should explain the forward fill issue");
        Assert.That(ex.Message, Does.Contain("Consider using FillNull()"),
            "Error message should suggest alternative");
    }

    /// <summary>
    /// Test that FillNullForward on disposed column throws ObjectDisposedException
    /// **Validates: Requirements 6.5**
    /// </summary>
    [Test]
    public void FillNullForward_OnDisposedColumn_ShouldThrowObjectDisposedException()
    {
        var column = NivaraColumn<int>.CreateFromNullable(new int?[] { 1, null, 3 });
        column.Dispose();
        
        Assert.Throws<ObjectDisposedException>(() => column.FillNullForward(),
            "FillNullForward on disposed column should throw ObjectDisposedException");
    }

    #endregion

    #region FillNullBackward Error Conditions

    /// <summary>
    /// Test that FillNullBackward throws InvalidOperationException when last element is null
    /// **Validates: Requirements 6.5**
    /// </summary>
    [Test]
    public void FillNullBackward_WithLastElementNull_ShouldThrowInvalidOperationException()
    {
        var column = NivaraColumn<int>.CreateFromNullable(new int?[] { 1, 2, null });
        
        var ex = Assert.Throws<InvalidOperationException>(() => column.FillNullBackward(),
            "FillNullBackward with last element null should throw InvalidOperationException");
        
        Assert.That(ex.Message, Does.Contain("no following non-null value"),
            "Error message should explain the backward fill issue");
        Assert.That(ex.Message, Does.Contain("Consider using FillNull()"),
            "Error message should suggest alternative");
    }

    /// <summary>
    /// Test that FillNullBackward on disposed column throws ObjectDisposedException
    /// **Validates: Requirements 6.5**
    /// </summary>
    [Test]
    public void FillNullBackward_OnDisposedColumn_ShouldThrowObjectDisposedException()
    {
        var column = NivaraColumn<int>.CreateFromNullable(new int?[] { 1, null, 3 });
        column.Dispose();
        
        Assert.Throws<ObjectDisposedException>(() => column.FillNullBackward(),
            "FillNullBackward on disposed column should throw ObjectDisposedException");
    }

    #endregion

    #region Null Checking Method Error Conditions

    /// <summary>
    /// Test that IsNull with out-of-bounds index throws IndexOutOfRangeException
    /// **Validates: Requirements 6.5**
    /// </summary>
    [Test]
    public void IsNull_WithOutOfBoundsIndex_ShouldThrowIndexOutOfRangeException()
    {
        var column = NivaraColumn<int>.CreateFromNullable(new int?[] { 1, null, 3 });
        
        // Test negative index
        Assert.Throws<IndexOutOfRangeException>(() => column.IsNull(-1),
            "IsNull with negative index should throw IndexOutOfRangeException");
        
        // Test index equal to length
        Assert.Throws<IndexOutOfRangeException>(() => column.IsNull(3),
            "IsNull with index equal to length should throw IndexOutOfRangeException");
        
        // Test index greater than length
        Assert.Throws<IndexOutOfRangeException>(() => column.IsNull(10),
            "IsNull with index greater than length should throw IndexOutOfRangeException");
    }

    /// <summary>
    /// Test that IsNull on disposed column throws ObjectDisposedException
    /// **Validates: Requirements 6.5**
    /// </summary>
    [Test]
    public void IsNull_OnDisposedColumn_ShouldThrowObjectDisposedException()
    {
        var column = NivaraColumn<int>.CreateFromNullable(new int?[] { 1, null, 3 });
        column.Dispose();
        
        Assert.Throws<ObjectDisposedException>(() => column.IsNull(0),
            "IsNull on disposed column should throw ObjectDisposedException");
    }

    /// <summary>
    /// Test that HasNulls on disposed column throws ObjectDisposedException
    /// **Validates: Requirements 6.5**
    /// </summary>
    [Test]
    public void HasNulls_OnDisposedColumn_ShouldThrowObjectDisposedException()
    {
        var column = NivaraColumn<int>.CreateFromNullable(new int?[] { 1, null, 3 });
        column.Dispose();
        
        Assert.Throws<ObjectDisposedException>(() => { var _ = column.HasNulls; },
            "HasNulls on disposed column should throw ObjectDisposedException");
    }

    /// <summary>
    /// Test that NullCount on disposed column throws ObjectDisposedException
    /// **Validates: Requirements 6.5**
    /// </summary>
    [Test]
    public void NullCount_OnDisposedColumn_ShouldThrowObjectDisposedException()
    {
        var column = NivaraColumn<int>.CreateFromNullable(new int?[] { 1, null, 3 });
        column.Dispose();
        
        Assert.Throws<ObjectDisposedException>(() => { var _ = column.NullCount; },
            "NullCount on disposed column should throw ObjectDisposedException");
    }

    /// <summary>
    /// Test that GetNullIndices on disposed column throws ObjectDisposedException
    /// **Validates: Requirements 6.5**
    /// </summary>
    [Test]
    public void GetNullIndices_OnDisposedColumn_ShouldThrowObjectDisposedException()
    {
        var column = NivaraColumn<int>.CreateFromNullable(new int?[] { 1, null, 3 });
        column.Dispose();
        
        Assert.Throws<ObjectDisposedException>(() => column.GetNullIndices(),
            "GetNullIndices on disposed column should throw ObjectDisposedException");
    }

    #endregion

    #region CreateFromNullable Error Conditions

    /// <summary>
    /// Test that CreateFromNullable with null array throws ArgumentNullException
    /// **Validates: Requirements 6.5**
    /// </summary>
    [Test]
    public void CreateFromNullable_WithNullArray_ShouldThrowArgumentNullException()
    {
        var ex = Assert.Throws<ArgumentNullException>(() => NivaraColumn<int>.CreateFromNullable(null!),
            "CreateFromNullable with null array should throw ArgumentNullException");
        
        Assert.That(ex.ParamName, Is.EqualTo("values"),
            "Exception should specify the values parameter");
    }

    /// <summary>
    /// Test that CreateFromNullable with non-value type throws InvalidOperationException
    /// **Validates: Requirements 6.5**
    /// </summary>
    [Test]
    public void CreateFromNullable_WithReferenceType_ShouldThrowInvalidOperationException()
    {
        var stringArray = new string?[] { "a", null, "c" };
        
        var ex = Assert.Throws<InvalidOperationException>(() => NivaraColumn<string>.CreateFromNullable(stringArray),
            "CreateFromNullable with reference type should throw InvalidOperationException");
        
        Assert.That(ex.Message, Does.Contain("can only be used with value types"),
            "Error message should explain value type requirement");
    }

    #endregion
}