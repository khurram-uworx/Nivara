using Microsoft.ML;
using Nivara.MLNet;
using NUnit.Framework;

namespace Nivara.Tests;

[TestFixture]
public class MLNetIntegrationTests
{
    private MLContext mlContext;

    [SetUp]
    public void Setup()
    {
        mlContext = new MLContext(seed: 42);
    }

    [Test]
    public void MLNetTensorRoundTrip_PreservesData()
    {
        var testData = new[] { 1.0f, 2.5f, -3.7f, 0.0f, 42.1f };

        // Create a NivaraSeries from the data
        var originalSeries = NivaraSeries<float>.Create(testData);

        // Convert to ML.NET tensor
        var mlNetTensor = originalSeries.ToMLNetTensor();

        // Convert back to NivaraSeries
        var roundTripSeries = MLNetInterop.FromMLNetTensor(mlNetTensor);

        // Assert that data is preserved
        Assert.That(roundTripSeries.Length, Is.EqualTo(originalSeries.Length));

        for (int i = 0; i < originalSeries.Length; i++)
        {
            var originalValue = originalSeries[i];
            var roundTripValue = roundTripSeries[i];
            Assert.That(roundTripValue, Is.EqualTo(originalValue).Within(1e-6f));
        }
    }

    [Test]
    public void NivaraFrameToDataView_PreservesStructure()
    {
        var col1Data = new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        var col2Data = new[] { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f };

        // Create NivaraFrame
        var col1 = NivaraColumn<float>.Create(col1Data);
        var col2 = NivaraColumn<float>.Create(col2Data);
        var originalFrame = NivaraFrame.Create(
            ("Col1", col1),
            ("Col2", col2)
        );

        // Convert to ML.NET DataView
        var dataView = originalFrame.ToDataView(mlContext);

        // Convert back to NivaraFrame
        var roundTripFrame = MLNetInterop.ToNivaraFrame(dataView, mlContext);

        // Assert structure is preserved
        Assert.That(roundTripFrame.RowCount, Is.EqualTo(originalFrame.RowCount));
        Assert.That(roundTripFrame.ColumnCount, Is.EqualTo(originalFrame.ColumnCount));
        Assert.That(roundTripFrame.ColumnNames, Contains.Item("Col1"));
        Assert.That(roundTripFrame.ColumnNames, Contains.Item("Col2"));

        // Assert data is preserved (within floating point precision)
        for (int row = 0; row < originalFrame.RowCount; row++)
        {
            var originalCol1 = originalFrame.GetColumn<float>("Col1")[row];
            var roundTripCol1 = roundTripFrame.GetColumn<float>("Col1")[row];
            Assert.That(roundTripCol1, Is.EqualTo(originalCol1).Within(1e-6f));

            var originalCol2 = originalFrame.GetColumn<float>("Col2")[row];
            var roundTripCol2 = roundTripFrame.GetColumn<float>("Col2")[row];
            Assert.That(roundTripCol2, Is.EqualTo(originalCol2).Within(1e-6f));
        }
    }

    [Test]
    public void FeatureVectorConversion_PreservesData()
    {
        var data1 = new[] { 1.0f, 2.0f, 3.0f };
        var data2 = new[] { 4.0f, 5.0f, 6.0f };

        // Create NivaraFrame with feature columns
        var col1 = NivaraColumn<float>.Create(data1);
        var col2 = NivaraColumn<float>.Create(data2);
        var frame = NivaraFrame.Create(
            ("Feature1", col1),
            ("Feature2", col2)
        );

        // Convert to feature vectors
        var featureVectors = frame.ToFeatureVectors("Feature1", "Feature2");

        // Convert back to NivaraFrame
        var roundTripFrame = MLNetInterop.FromFeatureVectors(featureVectors, new[] { "Feature1", "Feature2" });

        // Assert data is preserved
        Assert.That(roundTripFrame.RowCount, Is.EqualTo(frame.RowCount));
        Assert.That(roundTripFrame.ColumnCount, Is.EqualTo(2));

        for (int row = 0; row < frame.RowCount; row++)
        {
            var originalFeature1 = frame.GetColumn<float>("Feature1")[row];
            var roundTripFeature1 = roundTripFrame.GetColumn<float>("Feature1")[row];
            Assert.That(roundTripFeature1, Is.EqualTo(originalFeature1).Within(1e-6f));

            var originalFeature2 = frame.GetColumn<float>("Feature2")[row];
            var roundTripFeature2 = roundTripFrame.GetColumn<float>("Feature2")[row];
            Assert.That(roundTripFeature2, Is.EqualTo(originalFeature2).Within(1e-6f));
        }
    }

    [Test]
    public void DenseTensorConversion_PreservesData()
    {
        // Create 2D tensor
        var tensor = new float[3, 2] { { 1.0f, 2.0f }, { 3.0f, 4.0f }, { 5.0f, 6.0f } };

        // Convert to NivaraFrame
        var frame = TensorConversions.FromDenseTensor(tensor);

        // Convert back to tensor
        var roundTripTensor = frame.ToDenseTensor();

        // Assert dimensions are preserved
        Assert.That(roundTripTensor.GetLength(0), Is.EqualTo(3));
        Assert.That(roundTripTensor.GetLength(1), Is.EqualTo(2));

        // Assert data is preserved
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                Assert.That(roundTripTensor[i, j], Is.EqualTo(tensor[i, j]).Within(1e-6f));
            }
        }
    }

    [Test]
    public void SparseTensorConversion_PreservesNonZeroValues()
    {
        var rowData = new[] { 1.0f, 0.0f, 3.5f, 0.0f, -2.1f };

        // Create a single-row NivaraFrame
        var columns = new List<(string Name, IColumn Column)>();
        for (int i = 0; i < rowData.Length; i++)
        {
            var col = NivaraColumn<float>.Create(new[] { rowData[i] });
            columns.Add(($"Col_{i}", col));
        }
        var frame = new NivaraFrame(columns);

        // Convert to sparse tensors
        var sparseTensors = frame.ToSparseTensors(threshold: 1e-6f);

        // Convert back to NivaraFrame
        var roundTripFrame = TensorConversions.FromSparseTensors(sparseTensors);

        // Assert structure is preserved
        Assert.That(roundTripFrame.RowCount, Is.EqualTo(1));
        Assert.That(roundTripFrame.ColumnCount, Is.EqualTo(rowData.Length));

        // Assert data is preserved (within threshold)
        for (int col = 0; col < rowData.Length; col++)
        {
            var originalValue = rowData[col];
            var roundTripValue = roundTripFrame.GetColumn<float>($"Col_{col}")[0];

            if (Math.Abs(originalValue) >= 1e-6f)
            {
                // Non-zero values should be preserved
                Assert.That(roundTripValue, Is.EqualTo(originalValue).Within(1e-6f));
            }
            else
            {
                // Values below threshold should be zero
                Assert.That(Math.Abs(roundTripValue), Is.LessThanOrEqualTo(1e-6f));
            }
        }
    }

    [Test]
    public void TrainTestSplit_PreservesDataIntegrity()
    {
        // Create test data
        var data = Enumerable.Range(1, 100).Select(i => (float)i).ToArray();
        var col = NivaraColumn<float>.Create(data);
        var frame = NivaraFrame.Create(("Values", col));

        // Split the data
        var (training, testing) = frame.TrainTestSplit(trainRatio: 0.8, randomSeed: 42);

        // Verify split proportions
        Assert.That(training.RowCount, Is.EqualTo(80));
        Assert.That(testing.RowCount, Is.EqualTo(20));

        // Verify no data loss
        var allTrainingValues = new HashSet<float>();
        var allTestingValues = new HashSet<float>();

        var trainingCol = training.GetColumn<float>("Values");
        for (int i = 0; i < training.RowCount; i++)
        {
            allTrainingValues.Add(trainingCol[i]);
        }

        var testingCol = testing.GetColumn<float>("Values");
        for (int i = 0; i < testing.RowCount; i++)
        {
            allTestingValues.Add(testingCol[i]);
        }

        // No overlap between training and testing
        var intersection = allTrainingValues.Intersect(allTestingValues);
        Assert.That(intersection, Is.Empty);

        // All original values should be present
        var allValues = allTrainingValues.Union(allTestingValues);
        Assert.That(allValues.Count(), Is.EqualTo(100));

        // Verify all original data is present using sequence equal instead of set equal
        var sortedAllValues = allValues.OrderBy(x => x).ToArray();
        var sortedExpected = data.OrderBy(x => x).ToArray();
        Assert.That(sortedAllValues, Is.EqualTo(sortedExpected));
    }

    [Test]
    public void Normalization_ProducesZeroMeanUnitVariance()
    {
        // Create test data with known statistics
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };
        var col = NivaraColumn<float>.Create(data);
        var frame = NivaraFrame.Create(("Values", col));

        // Normalize the data
        var normalizedFrame = frame.Normalize("Values");

        // Extract normalized values
        var normalizedValues = new float[normalizedFrame.RowCount];
        var normalizedCol = normalizedFrame.GetColumn<float>("Values");
        for (int i = 0; i < normalizedFrame.RowCount; i++)
        {
            normalizedValues[i] = normalizedCol[i];
        }

        // Verify zero mean (within floating point precision)
        var mean = normalizedValues.Average();
        Assert.That(mean, Is.EqualTo(0.0f).Within(1e-6f));

        // Verify unit variance
        var variance = normalizedValues.Select(x => Math.Pow(x - mean, 2)).Average();
        var stdDev = Math.Sqrt(variance);
        Assert.That(stdDev, Is.EqualTo(1.0f).Within(1e-6f));
    }

    [Test]
    public void ToBatchTensors_ConvertsSeriesToVBuffers()
    {
        // Create test series
        var series1 = NivaraSeries<float>.Create(new float[] { 1.0f, 2.0f, 3.0f });
        var series2 = NivaraSeries<float>.Create(new float[] { 4.0f, 5.0f, 6.0f });
        var series3 = NivaraSeries<float>.Create(new float[] { 7.0f, 8.0f, 9.0f });
        var seriesList = new[] { series1, series2, series3 };

        // Convert to VBuffers
        var vbuffers = seriesList.ToBatchTensors();

        // Verify structure
        Assert.That(vbuffers.Length, Is.EqualTo(3));
        Assert.That(vbuffers[0].Length, Is.EqualTo(3));
        Assert.That(vbuffers[1].Length, Is.EqualTo(3));
        Assert.That(vbuffers[2].Length, Is.EqualTo(3));

        // Verify all VBuffers are dense
        Assert.That(vbuffers[0].IsDense, Is.True);
        Assert.That(vbuffers[1].IsDense, Is.True);
        Assert.That(vbuffers[2].IsDense, Is.True);

        // Verify data
        var values1 = vbuffers[0].GetValues();
        var values2 = vbuffers[1].GetValues();
        var values3 = vbuffers[2].GetValues();

        Assert.That(values1.ToArray(), Is.EqualTo(new float[] { 1.0f, 2.0f, 3.0f }));
        Assert.That(values2.ToArray(), Is.EqualTo(new float[] { 4.0f, 5.0f, 6.0f }));
        Assert.That(values3.ToArray(), Is.EqualTo(new float[] { 7.0f, 8.0f, 9.0f }));
    }

    [Test]
    public void ToBatchTensors_HandlesEmptyCollection()
    {
        var emptySeries = Array.Empty<NivaraSeries<float>>();
        var vbuffers = emptySeries.ToBatchTensors();
        
        Assert.That(vbuffers, Is.Empty);
    }

    [Test]
    public void ToBatchTensors_ThrowsOnNullInput()
    {
        IEnumerable<NivaraSeries<float>> nullSeries = null!;
        Assert.Throws<ArgumentNullException>(() => nullSeries.ToBatchTensors());
    }

    [Test]
    public void FromBatchTensors_ConvertsDenseVBuffersToSeries()
    {
        // Create dense VBuffers
        var vbuffer1 = new Microsoft.ML.Data.VBuffer<float>(3, new float[] { 1.0f, 2.0f, 3.0f });
        var vbuffer2 = new Microsoft.ML.Data.VBuffer<float>(3, new float[] { 4.0f, 5.0f, 6.0f });
        var vbuffers = new[] { vbuffer1, vbuffer2 };

        // Convert to series
        var series = TensorConversions.FromBatchTensors(vbuffers);

        // Verify structure
        Assert.That(series.Length, Is.EqualTo(2));
        Assert.That(series[0].Length, Is.EqualTo(3));
        Assert.That(series[1].Length, Is.EqualTo(3));

        // Verify data
        Assert.That(series[0].Values[0], Is.EqualTo(1.0f));
        Assert.That(series[0].Values[1], Is.EqualTo(2.0f));
        Assert.That(series[0].Values[2], Is.EqualTo(3.0f));

        Assert.That(series[1].Values[0], Is.EqualTo(4.0f));
        Assert.That(series[1].Values[1], Is.EqualTo(5.0f));
        Assert.That(series[1].Values[2], Is.EqualTo(6.0f));
    }

    [Test]
    public void FromBatchTensors_ConvertsSparseVBuffersToSeries()
    {
        // Create sparse VBuffers (length=5, but only 2 non-zero values each)
        var vbuffer1 = new Microsoft.ML.Data.VBuffer<float>(5, 2, new float[] { 1.5f, 3.7f }, new int[] { 1, 3 });
        var vbuffer2 = new Microsoft.ML.Data.VBuffer<float>(5, 2, new float[] { 2.1f, 4.2f }, new int[] { 0, 4 });
        var vbuffers = new[] { vbuffer1, vbuffer2 };

        // Convert to series
        var series = TensorConversions.FromBatchTensors(vbuffers);

        // Verify structure
        Assert.That(series.Length, Is.EqualTo(2));
        Assert.That(series[0].Length, Is.EqualTo(5));
        Assert.That(series[1].Length, Is.EqualTo(5));

        // Verify sparse data for first series: [0, 1.5, 0, 3.7, 0]
        Assert.That(series[0].Values[0], Is.EqualTo(0.0f));
        Assert.That(series[0].Values[1], Is.EqualTo(1.5f));
        Assert.That(series[0].Values[2], Is.EqualTo(0.0f));
        Assert.That(series[0].Values[3], Is.EqualTo(3.7f));
        Assert.That(series[0].Values[4], Is.EqualTo(0.0f));

        // Verify sparse data for second series: [2.1, 0, 0, 0, 4.2]
        Assert.That(series[1].Values[0], Is.EqualTo(2.1f));
        Assert.That(series[1].Values[1], Is.EqualTo(0.0f));
        Assert.That(series[1].Values[2], Is.EqualTo(0.0f));
        Assert.That(series[1].Values[3], Is.EqualTo(0.0f));
        Assert.That(series[1].Values[4], Is.EqualTo(4.2f));
    }

    [Test]
    public void FromBatchTensors_HandlesEmptyArray()
    {
        var emptyVBuffers = Array.Empty<Microsoft.ML.Data.VBuffer<float>>();
        var series = TensorConversions.FromBatchTensors(emptyVBuffers);
        
        Assert.That(series, Is.Empty);
    }

    [Test]
    public void FromBatchTensors_ThrowsOnNullInput()
    {
        Microsoft.ML.Data.VBuffer<float>[] nullVBuffers = null!;
        Assert.Throws<ArgumentNullException>(() => TensorConversions.FromBatchTensors(nullVBuffers));
    }

    [Test]
    public void VBufferRoundTrip_PreservesData()
    {
        // Create original series with various numeric types
        var floatSeries1 = NivaraSeries<float>.Create(new float[] { 1.1f, 2.2f, 3.3f });
        var floatSeries2 = NivaraSeries<float>.Create(new float[] { -1.5f, 0.0f, 42.7f });
        var originalSeries = new[] { floatSeries1, floatSeries2 };

        // Round trip: Series -> VBuffer -> Series
        var vbuffers = originalSeries.ToBatchTensors();
        var reconstructedSeries = TensorConversions.FromBatchTensors(vbuffers);

        // Verify structure preservation
        Assert.That(reconstructedSeries.Length, Is.EqualTo(originalSeries.Length));
        
        for (int i = 0; i < originalSeries.Length; i++)
        {
            Assert.That(reconstructedSeries[i].Length, Is.EqualTo(originalSeries[i].Length));
            
            // Verify data preservation
            for (int j = 0; j < originalSeries[i].Length; j++)
            {
                var original = originalSeries[i].Values[j];
                var reconstructed = reconstructedSeries[i].Values[j];
                Assert.That(reconstructed, Is.EqualTo(original).Within(1e-6f));
            }
        }
    }

    [Test]
    public void VBufferRoundTrip_WorksWithDifferentNumericTypes()
    {
        // Test with double type
        var doubleSeries = NivaraSeries<double>.Create(new double[] { 1.123456789, -2.987654321, 0.0 });
        var doubleSeriesArray = new[] { doubleSeries };

        var doubleVBuffers = doubleSeriesArray.ToBatchTensors();
        var reconstructedDoubleSeries = TensorConversions.FromBatchTensors(doubleVBuffers);

        Assert.That(reconstructedDoubleSeries.Length, Is.EqualTo(1));
        Assert.That(reconstructedDoubleSeries[0].Length, Is.EqualTo(3));
        
        for (int i = 0; i < 3; i++)
        {
            Assert.That(reconstructedDoubleSeries[0].Values[i], Is.EqualTo(doubleSeries.Values[i]).Within(1e-15));
        }

        // Test with int type
        var intSeries = NivaraSeries<int>.Create(new int[] { -100, 0, 42, 1000 });
        var intSeriesArray = new[] { intSeries };

        var intVBuffers = intSeriesArray.ToBatchTensors();
        var reconstructedIntSeries = TensorConversions.FromBatchTensors(intVBuffers);

        Assert.That(reconstructedIntSeries.Length, Is.EqualTo(1));
        Assert.That(reconstructedIntSeries[0].Length, Is.EqualTo(4));
        
        for (int i = 0; i < 4; i++)
        {
            Assert.That(reconstructedIntSeries[0].Values[i], Is.EqualTo(intSeries.Values[i]));
        }
    }

    [Test]
    public void ReshapeToTensor_CreatesCorrectDimensions()
    {
        // Create a series with 12 elements
        var data = Enumerable.Range(1, 12).Select(i => (float)i).ToArray();
        var series = NivaraSeries<float>.Create(data);

        // Reshape to 3x4 tensor
        var tensor = series.ReshapeToTensor(3, 4);

        // Verify dimensions
        Assert.That(tensor.Rank, Is.EqualTo(2));
        Assert.That(tensor.GetLength(0), Is.EqualTo(3));
        Assert.That(tensor.GetLength(1), Is.EqualTo(4));

        // Verify data mapping (row-major order)
        var expected = new float[,] {
            { 1f, 2f, 3f, 4f },
            { 5f, 6f, 7f, 8f },
            { 9f, 10f, 11f, 12f }
        };

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                Assert.That((float)tensor.GetValue(i, j)!, Is.EqualTo(expected[i, j]));
            }
        }
    }

    [Test]
    public void ReshapeToTensor_ThrowsOnDimensionMismatch()
    {
        var series = NivaraSeries<float>.Create(new float[] { 1f, 2f, 3f, 4f, 5f });

        // Try to reshape 5 elements into 2x3 (6 elements) - should fail
        Assert.Throws<ArgumentException>(() => series.ReshapeToTensor(2, 3));
    }

    [Test]
    public void FlattenFromTensor_ReconstructsOriginalSeries()
    {
        // Create a 2x3 tensor
        var tensor = new float[,] {
            { 1.1f, 2.2f, 3.3f },
            { 4.4f, 5.5f, 6.6f }
        };

        // Flatten to series
        var series = TensorConversions.FlattenFromTensor<float>(tensor);

        // Verify structure
        Assert.That(series.Length, Is.EqualTo(6));

        // Verify data (row-major flattening)
        var expectedValues = new float[] { 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f };
        for (int i = 0; i < expectedValues.Length; i++)
        {
            Assert.That(series.Values[i], Is.EqualTo(expectedValues[i]).Within(1e-6f));
        }
    }

    [Test]
    public void TensorReshapeRoundTrip_PreservesData()
    {
        // Create original series
        var originalData = new double[] { 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8 };
        var originalSeries = NivaraSeries<double>.Create(originalData);

        // Round trip: Series -> Tensor -> Series
        var tensor = originalSeries.ReshapeToTensor(2, 4);
        var reconstructedSeries = TensorConversions.FlattenFromTensor<double>(tensor);

        // Verify data preservation
        Assert.That(reconstructedSeries.Length, Is.EqualTo(originalSeries.Length));
        
        for (int i = 0; i < originalSeries.Length; i++)
        {
            Assert.That(reconstructedSeries.Values[i], Is.EqualTo(originalSeries.Values[i]).Within(1e-15));
        }
    }
}
