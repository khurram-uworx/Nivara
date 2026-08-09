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
    public void ReverseGradTensor_Half_IsSupported()
    {
        // Arrange
        var data = new Half[] { (Half)1.0, (Half)2.0, (Half)3.0 };
        var column = NivaraColumn<Half>.Create(data);

        // Act & Assert - Should not throw
        var tensor = new ReverseGradTensor<Half>(column, requiresGrad: true);
        Assert.That(tensor, Is.Not.Null);
        Assert.That(tensor.Length, Is.EqualTo(3));
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
    public void TypeValidator_IsSupported_Half_ReturnsTrue()
    {
        // Act
        var isSupported = TypeValidator.IsSupported<Half>();

        // Assert
        Assert.That(isSupported, Is.True);
    }

    [Test]
    public void TypeValidator_GetSupportedTypes_ReturnsFloatDoubleAndHalf()
    {
        // Act
        var supportedTypes = TypeValidator.GetSupportedTypes();

        // Assert
        Assert.That(supportedTypes, Has.Length.EqualTo(3));
        Assert.That(supportedTypes, Does.Contain(typeof(float)));
        Assert.That(supportedTypes, Does.Contain(typeof(double)));
        Assert.That(supportedTypes, Does.Contain(typeof(Half)));
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
    public void TypeConverter_ToHalf_ConvertsCorrectly()
    {
        // Arrange
        var data = new float[] { 1.5f, 2.5f, 3.5f };
        var column = NivaraColumn<float>.Create(data);
        var floatTensor = new ReverseGradTensor<float>(column, requiresGrad: true);

        // Act
        var halfTensor = TypeConverter.ToHalf(floatTensor);

        // Assert
        Assert.That(halfTensor, Is.Not.Null);
        Assert.That(halfTensor.Length, Is.EqualTo(3));
        Assert.That((double)halfTensor[0], Is.EqualTo(1.5).Within(0.0001));
        Assert.That((double)halfTensor[1], Is.EqualTo(2.5).Within(0.0001));
        Assert.That((double)halfTensor[2], Is.EqualTo(3.5).Within(0.0001));
        Assert.That(halfTensor.RequiresGrad, Is.True);
    }

    [Test]
    public void TypeConverter_ToHalf_FromDouble_PreservesRequiresGrad()
    {
        // Arrange
        var data = new double[] { 1.5, 2.5 };
        var column = NivaraColumn<double>.Create(data);
        var doubleTensor = new ReverseGradTensor<double>(column, requiresGrad: false);

        // Act
        var halfTensor = TypeConverter.ToHalf(doubleTensor);

        // Assert
        Assert.That(halfTensor.RequiresGrad, Is.False);
        Assert.That((double)halfTensor[0], Is.EqualTo(1.5).Within(0.0001));
    }

    [Test]
    public void ReverseGradTensor_ToHalf_ConvertsCorrectly()
    {
        // Arrange
        var data = new double[] { 1.5, 2.5, 3.5 };
        var column = NivaraColumn<double>.Create(data);
        var doubleTensor = new ReverseGradTensor<double>(column, requiresGrad: true);

        // Act
        var halfTensor = doubleTensor.ToHalf();

        // Assert
        Assert.That(halfTensor, Is.Not.Null);
        Assert.That(halfTensor.Length, Is.EqualTo(3));
        Assert.That((double)halfTensor[0], Is.EqualTo(1.5).Within(0.0001));
        Assert.That(halfTensor.RequiresGrad, Is.True);
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
    public void IsAutoGradSupported_Half_ReturnsTrue()
    {
        // Act
        var isSupported = NivaraAutoGradExtensions.IsAutoGradSupported<Half>();

        // Assert
        Assert.That(isSupported, Is.True);
    }

    [Test]
    public void GetSupportedAutoGradTypes_ReturnsCorrectTypes()
    {
        // Act
        var types = NivaraAutoGradExtensions.GetSupportedAutoGradTypes();

        // Assert
        Assert.That(types, Has.Length.EqualTo(3));
        Assert.That(types, Does.Contain(typeof(float)));
        Assert.That(types, Does.Contain(typeof(double)));
        Assert.That(types, Does.Contain(typeof(Half)));
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

    [Test]
    public void ReverseGradTensor_FromColumn_Nullable_ThrowsRuntime()
    {
        // Arrange
        var nullableCol = NivaraColumn<float>.CreateFromNullable(new float?[] { 1f, null, 3f });

        // Act & Assert
        var ex = Assert.Throws<AutoGradException>(() => ReverseGradTensor<float>.FromColumn(nullableCol));
        Assert.That(ex!.Message, Does.Contain("ADR-001"));
    }

    [Test]
    public void ReverseGradTensor_Constructor_Nullable_ThrowsRuntime()
    {
        // Arrange
        var nullableCol = NivaraColumn<float>.CreateFromNullable(new float?[] { 1f, null, 3f });

        // Act & Assert
        var ex = Assert.Throws<AutoGradException>(() => new ReverseGradTensor<float>(nullableCol));
        Assert.That(ex!.Message, Does.Contain("ADR-001"));
    }

    [Test]
    public void ForwardGradTensor_FromColumn_Nullable_ThrowsRuntime()
    {
        // Arrange
        var nullableCol = NivaraColumn<float>.CreateFromNullable(new float?[] { 1f, null, 3f });

        // Act & Assert
        var ex = Assert.Throws<AutoGradException>(() => ForwardGradTensor<float>.FromColumn(nullableCol));
        Assert.That(ex!.Message, Does.Contain("ADR-001"));
    }

    [Test]
    public void ForwardGradTensor_FromColumn_NullableTangent_ThrowsRuntime()
    {
        // Arrange
        var col = NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f });
        var nullableTangent = NivaraColumn<float>.CreateFromNullable(new float?[] { 1f, null, 3f });

        // Act & Assert
        var ex = Assert.Throws<AutoGradException>(() =>
            ForwardGradTensor<float>.FromColumn(col, nullableTangent));
        Assert.That(ex!.Message, Does.Contain("ADR-001"));
    }

    [Test]
    public void ReverseGradTensor_FromColumn_NonNullable_IsZeroCopy()
    {
        // Arrange
        var column = NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f });

        // Act
        var tensor = ReverseGradTensor<float>.FromColumn(column);

        // Assert
        Assert.That(tensor.Data.TryGetSpan(out var tensorSpan), Is.True);
        Assert.That(column.TryGetSpan(out var columnSpan), Is.True);
        Assert.That(tensorSpan == columnSpan, Is.True,
            "FromColumn must wrap the column zero-copy (shared backing array)");
    }

    [Test]
    public void ReverseGradTensor_FromArray_AliasesSourceArray()
    {
        // Arrange
        var array = new float[] { 1f, 2f, 3f };

        // Act
        var tensor = ReverseGradTensor<float>.FromArray(array);

        // Assert
        Assert.That(tensor.Data.TryGetSpan(out var span), Is.True);
        Assert.That(span == array.AsSpan(), Is.True,
            "FromArray must wrap the caller's array zero-copy");
    }

    [Test]
    public void ReverseGradTensor_FromMatrix_AliasesSourceArray()
    {
        // Arrange
        var array = new float[] { 1f, 2f, 3f, 4f };

        // Act
        var tensor = ReverseGradTensor<float>.FromMatrix(array, rows: 2, cols: 2);

        // Assert
        Assert.That(tensor.Data.TryGetSpan(out var span), Is.True);
        Assert.That(span == array.AsSpan(), Is.True,
            "FromMatrix must wrap the caller's array zero-copy");
        Assert.That(tensor.Shape, Is.EqualTo(new[] { 2, 2 }));
    }

    [Test]
    public void GradTensor_AsTensor_IsZeroCopyView()
    {
        // Arrange
        var column = NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f });
        var tensor = ReverseGradTensor<float>.FromColumn(column);

        // Act
        var tensorView = tensor.AsTensor();

        // Assert
        Assert.That(tensorView.TryGetSpan(new nint[] { 0 }, (int)tensorView.FlattenedLength, out Span<float> viewSpan), Is.True);
        Assert.That(column.TryGetSpan(out var columnSpan), Is.True);
        Assert.That((ReadOnlySpan<float>)viewSpan == columnSpan, Is.True,
            "AsTensor() must return a zero-copy view sharing the column's backing array");
    }

    [Test]
    public void NivaraColumn_AsTensorView_IsZeroCopyView()
    {
        // Arrange
        var column = NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f });

        // Act
        var tensorView = column.AsTensorView();

        // Assert
        Assert.That(tensorView.TryGetSpan(new nint[] { 0 }, (int)tensorView.FlattenedLength, out Span<float> viewSpan), Is.True);
        Assert.That(column.TryGetSpan(out var columnSpan), Is.True);
        Assert.That((ReadOnlySpan<float>)viewSpan == columnSpan, Is.True,
            "AsTensorView() must return a zero-copy view sharing the column's backing array");
    }

    [Test]
    public void NivaraColumn_AsTensorView_ThrowsOnNulls()
    {
        // Arrange
        var column = NivaraColumn<float>.CreateFromNullable(new float?[] { 1f, null, 3f });

        // Act + Assert
        var ex = Assert.Throws<InvalidOperationException>(() => column.AsTensorView());
        Assert.That(ex!.Message, Does.Contain("null"));
    }

    [Test]
    public void NivaraColumn_AsTensorView_ThrowsForReferenceTypes()
    {
        // Arrange
        var column = NivaraColumn<string>.CreateForReferenceType(new string[] { "a", "b", "c" });

        // Act + Assert
        Assert.Throws<InvalidOperationException>(() => column.AsTensorView());
    }

    [Test]
    public void NivaraSeries_AsTensorView_IsZeroCopyView()
    {
        // Arrange
        var series = new NivaraSeries<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f }));

        // Act
        var tensorView = series.AsTensorView();

        // Assert
        Assert.That(tensorView.TryGetSpan(new nint[] { 0 }, (int)tensorView.FlattenedLength, out Span<float> viewSpan), Is.True);
        Assert.That(series.Values.TryGetSpan(out var columnSpan), Is.True);
        Assert.That((ReadOnlySpan<float>)viewSpan == columnSpan, Is.True,
            "NivaraSeries.AsTensorView() must share the underlying column's backing array");
    }

}
