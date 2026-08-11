using Nivara.AutoDiff.Exceptions;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Tests for the AutoGrad exception hierarchy to ensure proper error handling and context reporting
/// </summary>
[TestFixture]
public class ExceptionTests
{
    [Test]
    public void AutoGradException_BasicConstructor_CreatesExceptionWithMessage()
    {
        // Arrange
        var message = "Test error message";

        // Act
        var exception = new AutoGradException(message);

        // Assert
        Assert.That(exception.Message, Is.EqualTo(message));
        Assert.That(exception.OperationContext, Is.Null);
        Assert.That(exception.InvolvedShapes, Is.Null);
    }

    [Test]
    public void AutoGradException_WithInnerException_PreservesInnerException()
    {
        // Arrange
        var innerException = new InvalidOperationException("Inner error");
        var message = "Outer error";

        // Act
        var exception = new AutoGradException(message, innerException);

        // Assert
        Assert.That(exception.Message, Is.EqualTo(message));
        Assert.That(exception.InnerException, Is.SameAs(innerException));
    }

    [Test]
    public void AutoGradException_GetDetailedContext_ReturnsFormattedInformation()
    {
        // Arrange
        var message = "Test error";
        var exception = new AutoGradException(message);

        // Act
        var context = exception.GetDetailedContext();

        // Assert
        Assert.That(context, Does.Contain("AutoGradException"));
        Assert.That(context, Does.Contain(message));
    }

    [Test]
    public void ShapeIncompatibilityException_WithShapeDetails_StoresShapeInfo()
    {
        // Arrange
        var message = "Shape mismatch";
        var operation = "Add";
        var expectedShape = new[] { 2, 3 };
        var actualShape = new[] { 3, 2 };

        // Act
        var exception = new ShapeIncompatibilityException(message, operation, expectedShape, actualShape);

        // Assert
        Assert.That(exception.Message, Is.EqualTo(message));
        Assert.That(exception.OperationContext, Is.EqualTo(operation));
        Assert.That(exception.ExpectedShape, Is.EqualTo(expectedShape));
        Assert.That(exception.ActualShape, Is.EqualTo(actualShape));
        Assert.That(exception.InvolvedShapes, Is.Not.Null);
        Assert.That(exception.InvolvedShapes!.Count, Is.EqualTo(2));
    }

    [Test]
    public void ShapeIncompatibilityException_GetDetailedContext_ShowsShapeDifferences()
    {
        // Arrange
        var message = "Shape mismatch";
        var operation = "MatMul";
        var expectedShape = new[] { 2, 3, 4 };
        var actualShape = new[] { 2, 4, 4 };

        // Act
        var exception = new ShapeIncompatibilityException(message, operation, expectedShape, actualShape);
        var context = exception.GetDetailedContext();

        // Assert
        Assert.That(context, Does.Contain("ShapeIncompatibilityException"));
        Assert.That(context, Does.Contain(message));
        Assert.That(context, Does.Contain("Expected Shape"));
        Assert.That(context, Does.Contain("Actual Shape"));
        Assert.That(context, Does.Contain("Shape Differences"));
        Assert.That(context, Does.Contain("Dimension 1")); // Should show the dimension that differs
    }

    [Test]
    public void ShapeIncompatibilityException_DifferentDimensionCount_ShowsDimensionMismatch()
    {
        // Arrange
        var message = "Dimension count mismatch";
        var operation = "Reshape";
        var expectedShape = new[] { 2, 3 };
        var actualShape = new[] { 2, 3, 1 };

        // Act
        var exception = new ShapeIncompatibilityException(message, operation, expectedShape, actualShape);
        var context = exception.GetDetailedContext();

        // Assert
        Assert.That(context, Does.Contain("Dimension count mismatch"));
        Assert.That(context, Does.Contain("expected 2"));
        Assert.That(context, Does.Contain("got 3"));
    }
}
