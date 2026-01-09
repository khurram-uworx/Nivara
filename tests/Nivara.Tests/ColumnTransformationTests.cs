using NUnit.Framework;

namespace Nivara.Tests;

[TestFixture]
public class ColumnTransformationTests
{
    [Test]
    public void Transform_WithSimpleFunction_ShouldTransformValues()
    {
        // Arrange
        var data = new[] { 1, 2, 3, 4, 5 };
        var column = NivaraColumn<int>.Create(data);

        // Act
        var transformed = column.Transform(x => x * 2);

        // Assert
        Assert.That(transformed.Length, Is.EqualTo(5));
        Assert.That(transformed[0], Is.EqualTo(2));
        Assert.That(transformed[1], Is.EqualTo(4));
        Assert.That(transformed[2], Is.EqualTo(6));
        Assert.That(transformed[3], Is.EqualTo(8));
        Assert.That(transformed[4], Is.EqualTo(10));
        Assert.That(transformed.HasNulls, Is.False);
    }

    [Test]
    public void Transform_WithNullValues_ShouldPreserveNulls()
    {
        // Arrange
        var nullableData = new int?[] { 1, null, 3, null, 5 };
        var column = NivaraColumn<int>.CreateFromNullable(nullableData);

        // Act
        var transformed = column.Transform(x => x * 2);

        // Assert
        Assert.That(transformed.Length, Is.EqualTo(5));
        Assert.That(transformed[0], Is.EqualTo(2));
        Assert.That(transformed.IsNull(1), Is.True);
        Assert.That(transformed[2], Is.EqualTo(6));
        Assert.That(transformed.IsNull(3), Is.True);
        Assert.That(transformed[4], Is.EqualTo(10));
        Assert.That(transformed.HasNulls, Is.True);
        Assert.That(transformed.NullCount, Is.EqualTo(2));
    }

    [Test]
    public void Transform_WithTypeChange_ShouldCreateCorrectType()
    {
        // Arrange
        var data = new[] { 1, 2, 3 };
        var column = NivaraColumn<int>.Create(data);

        // Act
        var transformed = column.Transform(x => x.ToString());

        // Assert
        Assert.That(transformed.Length, Is.EqualTo(3));
        Assert.That(transformed[0], Is.EqualTo("1"));
        Assert.That(transformed[1], Is.EqualTo("2"));
        Assert.That(transformed[2], Is.EqualTo("3"));
        Assert.That(transformed.ElementType, Is.EqualTo(typeof(string)));
    }

    [Test]
    public void TransformNonNull_WithNullValues_ShouldOnlyTransformNonNulls()
    {
        // Arrange
        var nullableData = new int?[] { 1, null, 3, null, 5 };
        var column = NivaraColumn<int>.CreateFromNullable(nullableData);

        // Act
        var transformed = column.TransformNonNull(x => x * 2);

        // Assert
        Assert.That(transformed.Length, Is.EqualTo(5));
        Assert.That(transformed[0], Is.EqualTo(2));
        Assert.That(transformed.IsNull(1), Is.True);
        Assert.That(transformed[2], Is.EqualTo(6));
        Assert.That(transformed.IsNull(3), Is.True);
        Assert.That(transformed[4], Is.EqualTo(10));
        Assert.That(transformed.HasNulls, Is.True);
        Assert.That(transformed.NullCount, Is.EqualTo(2));
    }

    [Test]
    public void Transform_WithException_ShouldThrowInvalidOperationException()
    {
        // Arrange
        var data = new[] { 1, 0, 3 }; // 0 will cause division by zero
        var column = NivaraColumn<int>.Create(data);

        // Act & Assert
        var ex = Assert.Throws<InvalidOperationException>(() => 
            column.Transform(x => 10 / x));

        Assert.That(ex.Message, Does.Contain("Transformation function threw an exception at index 1"));
        Assert.That(ex.InnerException, Is.TypeOf<DivideByZeroException>());
    }

    [TearDown]
    public void TearDown()
    {
        // Clean up any resources if needed
        GC.Collect();
        GC.WaitForPendingFinalizers();
    }
}