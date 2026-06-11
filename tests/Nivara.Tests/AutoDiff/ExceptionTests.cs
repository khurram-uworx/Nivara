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
    public void GradientComputationException_WithOperationDetails_StoresOperationInfo()
    {
        // Arrange
        var message = "Gradient computation failed";
        var operation = "MatMul";
        var operationContext = "Forward pass completed, backward failed";

        // Act
        var exception = new GradientComputationException(message, operation, operationContext);

        // Assert
        Assert.That(exception.Message, Is.EqualTo(message));
        Assert.That(exception.FailedOperation, Is.EqualTo(operation));
        Assert.That(exception.OperationContext, Is.EqualTo(operationContext));
    }

    [Test]
    public void GradientComputationException_GetDetailedContext_IncludesOperationInfo()
    {
        // Arrange
        var message = "Gradient computation failed";
        var operation = "MatMul";
        var exception = new GradientComputationException(message, operation);

        // Act
        var context = exception.GetDetailedContext();

        // Assert
        Assert.That(context, Does.Contain("GradientComputationException"));
        Assert.That(context, Does.Contain(message));
        Assert.That(context, Does.Contain(operation));
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

    [Test]
    public void CircularDependencyException_WithCyclePath_StoresCycleInfo()
    {
        // Arrange
        var message = "Circular dependency detected";
        var cycleOperations = new List<string> { "Add", "Multiply", "Subtract", "Add" };

        // Act
        var exception = new CircularDependencyException(message, cycleOperations);

        // Assert
        Assert.That(exception.Message, Is.EqualTo(message));
        Assert.That(exception.CycleOperationNames, Is.EqualTo(cycleOperations));
        Assert.That(exception.OperationContext, Is.EqualTo("Computation Graph Validation"));
    }

    [Test]
    public void CircularDependencyException_GetDetailedContext_ShowsCyclePath()
    {
        // Arrange
        var message = "Circular dependency detected";
        var cycleOperations = new List<string> { "Add", "Multiply", "Subtract" };
        var exception = new CircularDependencyException(message, cycleOperations);

        // Act
        var context = exception.GetDetailedContext();

        // Assert
        Assert.That(context, Does.Contain("CircularDependencyException"));
        Assert.That(context, Does.Contain(message));
        Assert.That(context, Does.Contain("Circular Dependency Path"));
        Assert.That(context, Does.Contain("Add"));
        Assert.That(context, Does.Contain("Multiply"));
        Assert.That(context, Does.Contain("Subtract"));
        Assert.That(context, Does.Contain("Back to: Add"));
    }

    [Test]
    public void InvalidBackwardCallException_WithTensorDetails_StoresTensorInfo()
    {
        // Arrange
        var message = "Cannot call backward on non-scalar tensor";
        var tensorShape = new[] { 2, 3 };
        var requiresGrad = true;

        // Act
        var exception = new InvalidBackwardCallException(message, tensorShape, requiresGrad);

        // Assert
        Assert.That(exception.Message, Is.EqualTo(message));
        Assert.That(exception.TensorShape, Is.EqualTo(tensorShape));
        Assert.That(exception.RequiresGrad, Is.EqualTo(requiresGrad));
        Assert.That(exception.OperationContext, Is.EqualTo("Backward Pass"));
    }

    [Test]
    public void InvalidBackwardCallException_GetDetailedContext_ProvidesGuidance()
    {
        // Arrange
        var message = "Cannot call backward on non-scalar tensor";
        var tensorShape = new[] { 2, 3 };
        var exception = new InvalidBackwardCallException(message, tensorShape, requiresGrad: true);

        // Act
        var context = exception.GetDetailedContext();

        // Assert
        Assert.That(context, Does.Contain("InvalidBackwardCallException"));
        Assert.That(context, Does.Contain(message));
        Assert.That(context, Does.Contain("Tensor Shape"));
        Assert.That(context, Does.Contain("Is Scalar: False"));
        Assert.That(context, Does.Contain("Guidance"));
        Assert.That(context, Does.Contain("backward() can only be called on scalar tensors"));
    }

    [Test]
    public void TypeValidationException_WithTypeDetails_StoresTypeInfo()
    {
        // Arrange
        var message = "Type mismatch";
        var operation = "Convert";
        var expectedType = typeof(float);
        var actualType = typeof(int);

        // Act
        var exception = new TypeValidationException(message, operation, expectedType, actualType);

        // Assert
        Assert.That(exception.Message, Is.EqualTo(message));
        Assert.That(exception.OperationContext, Is.EqualTo(operation));
        Assert.That(exception.ExpectedType, Is.EqualTo(expectedType));
        Assert.That(exception.ActualType, Is.EqualTo(actualType));
    }

    [Test]
    public void TypeValidationException_GetDetailedContext_ShowsTypeInformation()
    {
        // Arrange
        var message = "Type mismatch";
        var operation = "Cast";
        var expectedType = typeof(double);
        var actualType = typeof(string);
        var exception = new TypeValidationException(message, operation, expectedType, actualType);

        // Act
        var context = exception.GetDetailedContext();

        // Assert
        Assert.That(context, Does.Contain("TypeValidationException"));
        Assert.That(context, Does.Contain(message));
        Assert.That(context, Does.Contain("Expected Type: Double"));
        Assert.That(context, Does.Contain("Actual Type: String"));
    }

    [Test]
    public void ExceptionHierarchy_AllExceptionsInheritFromAutoGradException()
    {
        // Assert
        Assert.That(typeof(GradientComputationException).IsSubclassOf(typeof(AutoGradException)), Is.True);
        Assert.That(typeof(ShapeIncompatibilityException).IsSubclassOf(typeof(AutoGradException)), Is.True);
        Assert.That(typeof(CircularDependencyException).IsSubclassOf(typeof(AutoGradException)), Is.True);
        Assert.That(typeof(InvalidBackwardCallException).IsSubclassOf(typeof(AutoGradException)), Is.True);
        Assert.That(typeof(TypeValidationException).IsSubclassOf(typeof(AutoGradException)), Is.True);
    }

    [Test]
    public void ExceptionChaining_InnerExceptionPreserved()
    {
        // Arrange
        var innerException = new DivideByZeroException("Division by zero");
        var message = "Gradient computation failed due to division by zero";
        var operation = "Divide";

        // Act
        var exception = new GradientComputationException(message, operation, "Backward pass", innerException);

        // Assert
        Assert.That(exception.InnerException, Is.SameAs(innerException));
        var context = exception.GetDetailedContext();
        Assert.That(context, Does.Contain("Inner Exception"));
        Assert.That(context, Does.Contain("DivideByZeroException"));
    }
}
