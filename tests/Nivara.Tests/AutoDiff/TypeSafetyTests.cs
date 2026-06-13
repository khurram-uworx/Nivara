using Nivara.AutoDiff;
using Nivara.AutoDiff.Exceptions;
using Nivara.AutoDiff.Extensions;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Tests for type safety and validation in automatic differentiation.
/// Validates Requirements 9.1, 9.2, 9.3, 9.5
/// </summary>
[TestFixture]
public class TypeSafetyTests
{
    [Test]
    public void ReverseGradTensor_Float_IsSupported()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f };
        var column = NivaraColumn<float>.Create(data);

        // Act & Assert - Should not throw
        var tensor = new ReverseGradTensor<float>(column, requiresGrad: true);
        Assert.That(tensor, Is.Not.Null);
        Assert.That(tensor.Length, Is.EqualTo(3));
    }

    [Test]
    public void ReverseGradTensor_Double_IsSupported()
    {
        // Arrange
        var data = new double[] { 1.0, 2.0, 3.0 };
        var column = NivaraColumn<double>.Create(data);

        // Act & Assert - Should not throw
        var tensor = new ReverseGradTensor<double>(column, requiresGrad: true);
        Assert.That(tensor, Is.Not.Null);
        Assert.That(tensor.Length, Is.EqualTo(3));
    }

    [Test]
    public void ReverseGradTensor_Int_ThrowsTypeValidationException()
    {
        // Arrange
        var data = new int[] { 1, 2, 3 };
        var column = NivaraColumn<int>.Create(data);

        // Act & Assert
        var ex = Assert.Throws<TypeValidationException>(() => new ReverseGradTensor<int>(column, requiresGrad: true));
        Assert.That(ex.Message, Does.Contain("not supported"));
        Assert.That(ex.Message, Does.Contain("float, double"));
        Assert.That(ex.ExpectedType, Is.Not.Null);
        Assert.That(ex.ActualType, Is.EqualTo(typeof(int)));
    }

    [Test]
    public void ReverseGradTensor_Long_ThrowsTypeValidationException()
    {
        // Arrange
        var data = new long[] { 1L, 2L, 3L };
        var column = NivaraColumn<long>.Create(data);

        // Act & Assert
        var ex = Assert.Throws<TypeValidationException>(() => new ReverseGradTensor<long>(column, requiresGrad: true));
        Assert.That(ex.Message, Does.Contain("not supported"));
        Assert.That(ex.Message, Does.Contain("float, double"));
    }

    [Test]
    public void ReverseGradTensor_Decimal_ThrowsTypeValidationException()
    {
        // Arrange
        var data = new decimal[] { 1.0m, 2.0m, 3.0m };
        var column = NivaraColumn<decimal>.Create(data);

        // Act & Assert
        var ex = Assert.Throws<TypeValidationException>(() => new ReverseGradTensor<decimal>(column, requiresGrad: true));
        Assert.That(ex.Message, Does.Contain("not supported"));
        Assert.That(ex.Message, Does.Contain("float, double"));
    }

    [Test]
    public void TypeValidator_IsSupported_Float_ReturnsTrue()
    {
        // Act
        var isSupported = TypeValidator.IsSupported<float>();

        // Assert
        Assert.That(isSupported, Is.True);
    }

    [Test]
    public void TypeValidator_IsSupported_Double_ReturnsTrue()
    {
        // Act
        var isSupported = TypeValidator.IsSupported<double>();

        // Assert
        Assert.That(isSupported, Is.True);
    }

    [Test]
    public void TypeValidator_IsSupported_Int_ReturnsFalse()
    {
        // Act
        var isSupported = TypeValidator.IsSupported<int>();

        // Assert
        Assert.That(isSupported, Is.False);
    }

    [Test]
    public void TypeValidator_GetSupportedTypes_ReturnsFloatAndDouble()
    {
        // Act
        var supportedTypes = TypeValidator.GetSupportedTypes();

        // Assert
        Assert.That(supportedTypes, Has.Length.EqualTo(2));
        Assert.That(supportedTypes, Does.Contain(typeof(float)));
        Assert.That(supportedTypes, Does.Contain(typeof(double)));
    }

    [Test]
    public void TypeConverter_FloatToDouble_ConvertsCorrectly()
    {
        // Arrange
        var data = new float[] { 1.5f, 2.5f, 3.5f };
        var column = NivaraColumn<float>.Create(data);
        var floatTensor = new ReverseGradTensor<float>(column, requiresGrad: true);

        // Act
        var doubleTensor = TypeConverter.ToDouble(floatTensor);

        // Assert
        Assert.That(doubleTensor, Is.Not.Null);
        Assert.That(doubleTensor.Length, Is.EqualTo(3));
        Assert.That(doubleTensor[0], Is.EqualTo(1.5).Within(0.0001));
        Assert.That(doubleTensor[1], Is.EqualTo(2.5).Within(0.0001));
        Assert.That(doubleTensor[2], Is.EqualTo(3.5).Within(0.0001));
        Assert.That(doubleTensor.RequiresGrad, Is.True);
    }

    [Test]
    public void TypeConverter_DoubleToFloat_ConvertsCorrectly()
    {
        // Arrange
        var data = new double[] { 1.5, 2.5, 3.5 };
        var column = NivaraColumn<double>.Create(data);
        var doubleTensor = new ReverseGradTensor<double>(column, requiresGrad: true);

        // Act
        var floatTensor = TypeConverter.ToFloat(doubleTensor);

        // Assert
        Assert.That(floatTensor, Is.Not.Null);
        Assert.That(floatTensor.Length, Is.EqualTo(3));
        Assert.That(floatTensor[0], Is.EqualTo(1.5f).Within(0.0001f));
        Assert.That(floatTensor[1], Is.EqualTo(2.5f).Within(0.0001f));
        Assert.That(floatTensor[2], Is.EqualTo(3.5f).Within(0.0001f));
        Assert.That(floatTensor.RequiresGrad, Is.True);
    }

    [Test]
    public void TypeConverter_Convert_PreservesRequiresGrad()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f };
        var column = NivaraColumn<float>.Create(data);
        var floatTensor = new ReverseGradTensor<float>(column, requiresGrad: false);

        // Act
        var doubleTensor = TypeConverter.ToDouble(floatTensor);

        // Assert
        Assert.That(doubleTensor.RequiresGrad, Is.False);
    }

    [Test]
    public void TypeConverter_Convert_CanOverrideRequiresGrad()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f };
        var column = NivaraColumn<float>.Create(data);
        var floatTensor = new ReverseGradTensor<float>(column, requiresGrad: false);

        // Act
        var doubleTensor = TypeConverter.ToDouble(floatTensor, requiresGrad: true);

        // Assert
        Assert.That(doubleTensor.RequiresGrad, Is.True);
    }

    [Test]
    public void GradTensor_WithNullData_ThrowsAutoGradException()
    {
        var column = NivaraColumn<float>.CreateFromNullable(new float?[] { 1.5f, null, 3.5f });
        Assert.Throws<AutoGradException>(() => new ReverseGradTensor<float>(column));
    }

    [Test]
    public void ReverseGradTensor_ToFloat_ConvertsCorrectly()
    {
        // Arrange
        var data = new double[] { 1.5, 2.5, 3.5 };
        var column = NivaraColumn<double>.Create(data);
        var doubleTensor = new ReverseGradTensor<double>(column, requiresGrad: true);

        // Act
        var floatTensor = doubleTensor.ToFloat();

        // Assert
        Assert.That(floatTensor, Is.Not.Null);
        Assert.That(floatTensor.Length, Is.EqualTo(3));
        Assert.That(floatTensor[0], Is.EqualTo(1.5f).Within(0.0001f));
        Assert.That(floatTensor.RequiresGrad, Is.True);
    }

    [Test]
    public void ReverseGradTensor_ToDouble_ConvertsCorrectly()
    {
        // Arrange
        var data = new float[] { 1.5f, 2.5f, 3.5f };
        var column = NivaraColumn<float>.Create(data);
        var floatTensor = new ReverseGradTensor<float>(column, requiresGrad: true);

        // Act
        var doubleTensor = floatTensor.ToDouble();

        // Assert
        Assert.That(doubleTensor, Is.Not.Null);
        Assert.That(doubleTensor.Length, Is.EqualTo(3));
        Assert.That(doubleTensor[0], Is.EqualTo(1.5).Within(0.0001));
        Assert.That(doubleTensor.RequiresGrad, Is.True);
    }

    [Test]
    public void NivaraColumn_ToReverseGradTensor_Float_Works()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f };
        var column = NivaraColumn<float>.Create(data);

        // Act
        var tensor = column.ToReverseGradTensor(requiresGrad: true);

        // Assert
        Assert.That(tensor, Is.Not.Null);
        Assert.That(tensor.Length, Is.EqualTo(3));
        Assert.That(tensor.RequiresGrad, Is.True);
    }

    [Test]
    public void NivaraSeries_ToReverseGradTensor_Double_Works()
    {
        // Arrange
        var data = new double[] { 1.0, 2.0, 3.0 };
        var column = NivaraColumn<double>.Create(data);
        var series = new NivaraSeries<double>(column);

        // Act
        var tensor = series.ToReverseGradTensor(requiresGrad: true);

        // Assert
        Assert.That(tensor, Is.Not.Null);
        Assert.That(tensor.Length, Is.EqualTo(3));
        Assert.That(tensor.RequiresGrad, Is.True);
    }

    [Test]
    public void IsAutoGradSupported_Float_ReturnsTrue()
    {
        // Act
        var isSupported = NivaraAutoGradExtensions.IsAutoGradSupported<float>();

        // Assert
        Assert.That(isSupported, Is.True);
    }

    [Test]
    public void IsAutoGradSupported_Int_ReturnsFalse()
    {
        // Act
        var isSupported = NivaraAutoGradExtensions.IsAutoGradSupported<int>();

        // Assert
        Assert.That(isSupported, Is.False);
    }

    [Test]
    public void GetSupportedAutoGradTypes_ReturnsCorrectTypes()
    {
        // Act
        var types = NivaraAutoGradExtensions.GetSupportedAutoGradTypes();

        // Assert
        Assert.That(types, Has.Length.EqualTo(2));
        Assert.That(types, Does.Contain(typeof(float)));
        Assert.That(types, Does.Contain(typeof(double)));
    }

    [Test]
    public void TypeValidator_ValidateScalar_NonScalar_ThrowsException()
    {
        // Act & Assert
        var ex = Assert.Throws<AutoGradException>(() =>
            TypeValidator.ValidateScalar(5, "TestOperation"));
        Assert.That(ex.Message, Does.Contain("scalar"));
        Assert.That(ex.Message, Does.Contain("length=1"));
    }

    [Test]
    public void TypeValidator_ValidateScalar_Scalar_DoesNotThrow()
    {
        // Act & Assert - Should not throw
        Assert.DoesNotThrow(() => TypeValidator.ValidateScalar(1, "TestOperation"));
    }

    [Test]
    public void TypeValidator_ValidateNonEmpty_Empty_ThrowsException()
    {
        // Act & Assert
        var ex = Assert.Throws<AutoGradException>(() =>
            TypeValidator.ValidateNonEmpty(0, "TestOperation"));
        Assert.That(ex.Message, Does.Contain("empty"));
    }

    [Test]
    public void TypeValidator_ValidateNonEmpty_NonEmpty_DoesNotThrow()
    {
        // Act & Assert - Should not throw
        Assert.DoesNotThrow(() => TypeValidator.ValidateNonEmpty(5, "TestOperation"));
    }

    [Test]
    public void TypeValidator_ValidateShapeCompatibility_Incompatible_ThrowsException()
    {
        // Act & Assert
        var ex = Assert.Throws<ShapeIncompatibilityException>(() =>
            TypeValidator.ValidateShapeCompatibility(5, 3, "TestOperation"));
        Assert.That(ex.Message, Does.Contain("Shape mismatch"));
        Assert.That(ex.Message, Does.Contain("expected length 3"));
        Assert.That(ex.Message, Does.Contain("got 5"));
    }

    [Test]
    public void TypeValidator_ValidateShapeCompatibility_Compatible_DoesNotThrow()
    {
        // Act & Assert - Should not throw
        Assert.DoesNotThrow(() => TypeValidator.ValidateShapeCompatibility(5, 5, "TestOperation"));
    }

}
