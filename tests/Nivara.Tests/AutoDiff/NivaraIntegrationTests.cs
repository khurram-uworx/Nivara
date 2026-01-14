using Nivara.Extensions.AutoDiff;
using Nivara.Extensions.AutoDiff.Extensions;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Tests for Nivara integration with automatic differentiation.
/// Validates seamless conversion between Nivara types and GradTensors,
/// batch operations, and frame-level ML workflows.
/// </summary>
[TestFixture]
public class NivaraIntegrationTests
{
    [Test]
    public void ToGradTensor_FromColumn_PreservesData()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var column = NivaraColumn<float>.Create(data);

        // Act
        var gradTensor = column.ToGradTensor(requiresGrad: true);

        // Assert
        Assert.That(gradTensor.Length, Is.EqualTo(4));
        Assert.That(gradTensor.RequiresGrad, Is.True);
        Assert.That(gradTensor[0], Is.EqualTo(1.0f));
        Assert.That(gradTensor[1], Is.EqualTo(2.0f));
        Assert.That(gradTensor[2], Is.EqualTo(3.0f));
        Assert.That(gradTensor[3], Is.EqualTo(4.0f));
    }

    [Test]
    public void ToGradTensor_FromSeries_PreservesData()
    {
        // Arrange
        var data = new double[] { 1.0, 2.0, 3.0 };
        var series = NivaraSeries<double>.Create(data);

        // Act
        var gradTensor = series.ToGradTensor(requiresGrad: false);

        // Assert
        Assert.That(gradTensor.Length, Is.EqualTo(3));
        Assert.That(gradTensor.RequiresGrad, Is.False);
        Assert.That(gradTensor[0], Is.EqualTo(1.0));
        Assert.That(gradTensor[1], Is.EqualTo(2.0));
        Assert.That(gradTensor[2], Is.EqualTo(3.0));
    }

    [Test]
    public void ToGradTensor_WithNullColumn_ThrowsArgumentNullException()
    {
        // Arrange
        NivaraColumn<float>? column = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => column!.ToGradTensor());
    }

    [Test]
    public void ToGradTensor_WithNullSeries_ThrowsArgumentNullException()
    {
        // Arrange
        NivaraSeries<double>? series = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => series!.ToGradTensor());
    }

    [Test]
    public void ToColumn_FromGradTensor_PreservesData()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f };
        var gradTensor = GradTensor<float>.FromArray(data, requiresGrad: true);

        // Act
        var column = gradTensor.ToColumn();

        // Assert
        Assert.That(column.Length, Is.EqualTo(3));
        Assert.That(column[0], Is.EqualTo(1.0f));
        Assert.That(column[1], Is.EqualTo(2.0f));
        Assert.That(column[2], Is.EqualTo(3.0f));
    }

    [Test]
    public void ToSeries_FromGradTensor_PreservesData()
    {
        // Arrange
        var data = new double[] { 1.0, 2.0, 3.0 };
        var gradTensor = GradTensor<double>.FromArray(data, requiresGrad: false);

        // Act
        var series = gradTensor.ToSeries();

        // Assert
        Assert.That(series.Length, Is.EqualTo(3));
        Assert.That(series[0], Is.EqualTo(1.0));
        Assert.That(series[1], Is.EqualTo(2.0));
        Assert.That(series[2], Is.EqualTo(3.0));
    }

    [Test]
    public void ToGradTensors_FromFrame_ConvertsMultipleColumns()
    {
        // Arrange
        var col1 = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f });
        var col2 = NivaraColumn<float>.Create(new float[] { 4.0f, 5.0f, 6.0f });
        var frame = NivaraFrame.Create(("A", col1), ("B", col2));

        // Act
        var tensors = frame.ToGradTensors<float>(new[] { "A", "B" }, requiresGrad: true);

        // Assert
        Assert.That(tensors.Count, Is.EqualTo(2));
        Assert.That(tensors.ContainsKey("A"), Is.True);
        Assert.That(tensors.ContainsKey("B"), Is.True);
        Assert.That(tensors["A"].Length, Is.EqualTo(3));
        Assert.That(tensors["B"].Length, Is.EqualTo(3));
        Assert.That(tensors["A"].RequiresGrad, Is.True);
        Assert.That(tensors["B"].RequiresGrad, Is.True);
    }

    [Test]
    public void ToGradTensors_WithNullFrame_ThrowsArgumentNullException()
    {
        // Arrange
        NivaraFrame? frame = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            frame!.ToGradTensors<float>(new[] { "A" }));
    }

    [Test]
    public void ToGradTensors_WithNullColumnNames_ThrowsArgumentNullException()
    {
        // Arrange
        var col = NivaraColumn<float>.Create(new float[] { 1.0f });
        var frame = NivaraFrame.Create(("A", col));

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            frame.ToGradTensors<float>(null!));
    }

    [Test]
    public void ToGradTensors_WithEmptyColumnNames_ThrowsArgumentException()
    {
        // Arrange
        var col = NivaraColumn<float>.Create(new float[] { 1.0f });
        var frame = NivaraFrame.Create(("A", col));

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            frame.ToGradTensors<float>(Array.Empty<string>()));
    }

    [Test]
    public void ToGradTensorsAuto_ConvertsOnlySupportedTypes()
    {
        // Arrange
        var floatCol = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f });
        var doubleCol = NivaraColumn<double>.Create(new double[] { 3.0, 4.0 });
        var intCol = NivaraColumn<int>.Create(new int[] { 5, 6 });
        var frame = NivaraFrame.Create(
            ("Float", floatCol),
            ("Double", doubleCol),
            ("Int", intCol));

        // Act
        var tensors = frame.ToGradTensorsAuto(requiresGrad: true);

        // Assert
        Assert.That(tensors.Count, Is.EqualTo(2)); // Only float and double
        Assert.That(tensors.ContainsKey("Float"), Is.True);
        Assert.That(tensors.ContainsKey("Double"), Is.True);
        Assert.That(tensors.ContainsKey("Int"), Is.False);
    }

    [Test]
    public void BatchZeroGrad_ClearsAllGradients()
    {
        // Arrange
        var tensor1 = GradTensor<float>.FromArray(new float[] { 1.0f }, requiresGrad: true);
        var tensor2 = GradTensor<float>.FromArray(new float[] { 2.0f }, requiresGrad: true);

        // Set some gradients
        tensor1.Grad = NivaraColumn<float>.Create(new float[] { 0.5f });
        tensor2.Grad = NivaraColumn<float>.Create(new float[] { 0.3f });

        var tensors = new Dictionary<string, GradTensor<float>>
        {
            { "T1", tensor1 },
            { "T2", tensor2 }
        };

        // Act
        tensors.BatchZeroGrad();

        // Assert
        Assert.That(tensor1.Grad, Is.Null);
        Assert.That(tensor2.Grad, Is.Null);
    }

    [Test]
    public void BatchZeroGrad_WithNullDictionary_ThrowsArgumentNullException()
    {
        // Arrange
        Dictionary<string, GradTensor<float>>? tensors = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => tensors!.BatchZeroGrad());
    }

    [Test]
    public void ToFrame_ConvertsGradTensorsToFrame()
    {
        // Arrange
        var tensor1 = GradTensor<float>.FromArray(new float[] { 1.0f, 2.0f }, requiresGrad: true);
        var tensor2 = GradTensor<float>.FromArray(new float[] { 3.0f, 4.0f }, requiresGrad: true);

        var tensors = new Dictionary<string, GradTensor<float>>
        {
            { "A", tensor1 },
            { "B", tensor2 }
        };

        // Act
        var frame = tensors.ToFrame();

        // Assert
        Assert.That(frame.ColumnCount, Is.EqualTo(2));
        Assert.That(frame.RowCount, Is.EqualTo(2));
        Assert.That(frame.HasColumn("A"), Is.True);
        Assert.That(frame.HasColumn("B"), Is.True);

        var colA = frame.GetColumn<float>("A");
        var colB = frame.GetColumn<float>("B");
        Assert.That(colA[0], Is.EqualTo(1.0f));
        Assert.That(colA[1], Is.EqualTo(2.0f));
        Assert.That(colB[0], Is.EqualTo(3.0f));
        Assert.That(colB[1], Is.EqualTo(4.0f));
    }

    [Test]
    public void ToFrame_WithNullDictionary_ThrowsArgumentNullException()
    {
        // Arrange
        Dictionary<string, GradTensor<float>>? tensors = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => tensors!.ToFrame());
    }

    [Test]
    public void ToFrame_WithEmptyDictionary_ThrowsArgumentException()
    {
        // Arrange
        var tensors = new Dictionary<string, GradTensor<float>>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => tensors.ToFrame());
    }

    [Test]
    public void ToGradientFrame_ExtractsGradients()
    {
        // Arrange
        var tensor1 = GradTensor<float>.FromArray(new float[] { 1.0f, 2.0f }, requiresGrad: true);
        var tensor2 = GradTensor<float>.FromArray(new float[] { 3.0f, 4.0f }, requiresGrad: true);

        // Set gradients
        tensor1.Grad = NivaraColumn<float>.Create(new float[] { 0.1f, 0.2f });
        tensor2.Grad = NivaraColumn<float>.Create(new float[] { 0.3f, 0.4f });

        var tensors = new Dictionary<string, GradTensor<float>>
        {
            { "A", tensor1 },
            { "B", tensor2 }
        };

        // Act
        var gradFrame = tensors.ToGradientFrame();

        // Assert
        Assert.That(gradFrame, Is.Not.Null);
        Assert.That(gradFrame!.ColumnCount, Is.EqualTo(2));
        Assert.That(gradFrame.RowCount, Is.EqualTo(2));

        var gradA = gradFrame.GetColumn<float>("A");
        var gradB = gradFrame.GetColumn<float>("B");
        Assert.That(gradA[0], Is.EqualTo(0.1f));
        Assert.That(gradA[1], Is.EqualTo(0.2f));
        Assert.That(gradB[0], Is.EqualTo(0.3f));
        Assert.That(gradB[1], Is.EqualTo(0.4f));
    }

    [Test]
    public void ToGradientFrame_WithNoGradients_ReturnsNull()
    {
        // Arrange
        var tensor1 = GradTensor<float>.FromArray(new float[] { 1.0f }, requiresGrad: true);
        var tensor2 = GradTensor<float>.FromArray(new float[] { 2.0f }, requiresGrad: true);

        var tensors = new Dictionary<string, GradTensor<float>>
        {
            { "A", tensor1 },
            { "B", tensor2 }
        };

        // Act
        var gradFrame = tensors.ToGradientFrame();

        // Assert
        Assert.That(gradFrame, Is.Null);
    }

    [Test]
    public void ToGradientFrame_WithNullDictionary_ThrowsArgumentNullException()
    {
        // Arrange
        Dictionary<string, GradTensor<float>>? tensors = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => tensors!.ToGradientFrame());
    }

    [Test]
    public void ToGradientFrame_WithEmptyDictionary_ThrowsArgumentException()
    {
        // Arrange
        var tensors = new Dictionary<string, GradTensor<float>>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => tensors.ToGradientFrame());
    }

    [Test]
    public void RoundTrip_ColumnToGradTensorToColumn_PreservesData()
    {
        // Arrange
        var originalData = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        var originalColumn = NivaraColumn<float>.Create(originalData);

        // Act
        var gradTensor = originalColumn.ToGradTensor(requiresGrad: true);
        var resultColumn = gradTensor.ToColumn();

        // Assert
        Assert.That(resultColumn.Length, Is.EqualTo(originalColumn.Length));
        for (int i = 0; i < originalData.Length; i++)
        {
            Assert.That(resultColumn[i], Is.EqualTo(originalColumn[i]));
        }
    }

    [Test]
    public void RoundTrip_SeriesToGradTensorToSeries_PreservesData()
    {
        // Arrange
        var originalData = new double[] { 1.0, 2.0, 3.0 };
        var originalSeries = NivaraSeries<double>.Create(originalData);

        // Act
        var gradTensor = originalSeries.ToGradTensor(requiresGrad: false);
        var resultSeries = gradTensor.ToSeries();

        // Assert
        Assert.That(resultSeries.Length, Is.EqualTo(originalSeries.Length));
        for (int i = 0; i < originalData.Length; i++)
        {
            Assert.That(resultSeries[i], Is.EqualTo(originalSeries[i]));
        }
    }

    [Test]
    public void RoundTrip_FrameToGradTensorsToFrame_PreservesData()
    {
        // Arrange
        var col1 = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f });
        var col2 = NivaraColumn<float>.Create(new float[] { 4.0f, 5.0f, 6.0f });
        var originalFrame = NivaraFrame.Create(("A", col1), ("B", col2));

        // Act
        var tensors = originalFrame.ToGradTensors<float>(new[] { "A", "B" }, requiresGrad: true);
        var resultFrame = tensors.ToFrame();

        // Assert
        Assert.That(resultFrame.ColumnCount, Is.EqualTo(originalFrame.ColumnCount));
        Assert.That(resultFrame.RowCount, Is.EqualTo(originalFrame.RowCount));

        var resultCol1 = resultFrame.GetColumn<float>("A");
        var resultCol2 = resultFrame.GetColumn<float>("B");

        for (int i = 0; i < 3; i++)
        {
            Assert.That(resultCol1[i], Is.EqualTo(col1[i]));
            Assert.That(resultCol2[i], Is.EqualTo(col2[i]));
        }
    }

    [Test]
    public void IsAutoGradSupported_WithFloat_ReturnsTrue()
    {
        // Act
        var isSupported = NivaraAutoGradExtensions.IsAutoGradSupported<float>();

        // Assert
        Assert.That(isSupported, Is.True);
    }

    [Test]
    public void IsAutoGradSupported_WithDouble_ReturnsTrue()
    {
        // Act
        var isSupported = NivaraAutoGradExtensions.IsAutoGradSupported<double>();

        // Assert
        Assert.That(isSupported, Is.True);
    }

    [Test]
    public void IsAutoGradSupported_WithInt_ReturnsFalse()
    {
        // Act
        var isSupported = NivaraAutoGradExtensions.IsAutoGradSupported<int>();

        // Assert
        Assert.That(isSupported, Is.False);
    }

    [Test]
    public void GetSupportedAutoGradTypes_ReturnsFloatAndDouble()
    {
        // Act
        var supportedTypes = NivaraAutoGradExtensions.GetSupportedAutoGradTypes();

        // Assert
        Assert.That(supportedTypes, Is.Not.Null);
        Assert.That(supportedTypes.Length, Is.EqualTo(2));
        Assert.That(supportedTypes, Does.Contain(typeof(float)));
        Assert.That(supportedTypes, Does.Contain(typeof(double)));
    }

    [Test]
    public void NullHandling_ColumnWithNulls_PreservesNullSemantics()
    {
        // Arrange
        var nullableData = new float?[] { 1.0f, null, 3.0f, null, 5.0f };
        var column = NivaraColumn<float>.CreateFromNullable(nullableData);

        // Act
        var gradTensor = column.ToGradTensor(requiresGrad: true);

        // Assert
        Assert.That(gradTensor.HasNulls, Is.True);
        Assert.That(gradTensor.IsNull(1), Is.True);
        Assert.That(gradTensor.IsNull(3), Is.True);
        Assert.That(gradTensor.IsNull(0), Is.False);
        Assert.That(gradTensor.IsNull(2), Is.False);
        Assert.That(gradTensor.IsNull(4), Is.False);
    }

    [Test]
    public void NullHandling_RoundTripWithNulls_PreservesNullMask()
    {
        // Arrange
        var nullableData = new double?[] { 1.0, null, 3.0 };
        var originalColumn = NivaraColumn<double>.CreateFromNullable(nullableData);

        // Act
        var gradTensor = originalColumn.ToGradTensor(requiresGrad: false);
        var resultColumn = gradTensor.ToColumn();

        // Assert
        Assert.That(resultColumn.HasNulls, Is.True);
        Assert.That(resultColumn.IsNull(0), Is.False);
        Assert.That(resultColumn.IsNull(1), Is.True);
        Assert.That(resultColumn.IsNull(2), Is.False);
    }
}
