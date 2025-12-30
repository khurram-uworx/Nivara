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
        // Feature: dataframe-library, Property 10: ML.NET Integration Round-Trip
        // Validates: Requirements 3.4, 7.4
        
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
        // Feature: dataframe-library, Property 10: ML.NET Integration Round-Trip
        // Validates: Requirements 3.4, 7.4

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
        // Feature: dataframe-library, Property 10: ML.NET Integration Round-Trip
        // Validates: Requirements 3.4, 7.4

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
        // Feature: dataframe-library, Property 10: ML.NET Integration Round-Trip
        // Validates: Requirements 3.4, 7.4

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
        // Feature: dataframe-library, Property 10: ML.NET Integration Round-Trip
        // Validates: Requirements 3.4, 7.4

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
        // Feature: dataframe-library, Property 10: ML.NET Integration Round-Trip
        // Validates: Requirements 3.4, 7.4

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
        // Feature: dataframe-library, Property 10: ML.NET Integration Round-Trip
        // Validates: Requirements 3.4, 7.4

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
}
