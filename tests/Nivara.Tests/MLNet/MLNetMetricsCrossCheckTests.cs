using Microsoft.ML;
using Microsoft.ML.Data;
using Nivara.Metrics;
using NUnit.Framework;

namespace Nivara.Tests.MLNet;

[TestFixture]
public class MLNetMetricsCrossCheckTests
{
    [Test]
    public void ConfusionMatrix_MatchesMLNetEvaluation_Exactly()
    {
        // Deterministic, deliberately asymmetric 3-class confusion
        // (rows = predicted, columns = actual): 192 rows in total.
        int[,] confusion =
        {
            { 60, 10, 4 },
            { 3, 45, 8 },
            { 1, 6, 55 },
        };

        var predictedList = new List<int>();
        var actualList = new List<int>();
        for (int p = 0; p < 3; p++)
            for (int a = 0; a < 3; a++)
                for (int k = 0; k < confusion[p, a]; k++)
                {
                    predictedList.Add(p);
                    actualList.Add(a);
                }

        var predicted = NivaraColumn<int>.Create(predictedList.ToArray());
        var actual = NivaraColumn<int>.Create(actualList.ToArray());

        NivaraFrame frame = NivaraMetrics.ConfusionMatrixFrame(predicted, actual);

        // Build the predictions IDataView directly: no training needed for Evaluate.
        var mlContext = new MLContext(seed: 42);
        var rows = predictedList
            .Zip(actualList, (p, a) => new ClassificationRow
            {
                Label = (float)a,
                PredictedLabel = (float)p,
                Score = OneHot(p, 3),
            })
            .ToArray();
        var dataView = mlContext.Data.LoadFromEnumerable(rows);
        var ml = mlContext.MulticlassClassification.Evaluate(dataView, "Label", "Score", "PredictedLabel");

        int classes = frame.RowCount;
        Assert.That(classes, Is.EqualTo(3));

        // Nivara stores predicted in rows, actual in columns.
        // ML.NET ConfusionMatrix.Counts stores actual in rows, predicted in columns.
        for (int p = 0; p < classes; p++)
            for (int a = 0; a < classes; a++)
            {
                long nivaraCell = frame.GetColumn<long>($"Actual:{a}")[p];
                long mlCell = (long)ml.ConfusionMatrix.Counts[a][p];
                Assert.That(mlCell, Is.EqualTo(nivaraCell));
            }

        long totalCorrect = 0;
        long totalSupport = 0;
        for (int i = 0; i < classes; i++)
        {
            totalCorrect += frame.GetColumn<long>("Correct")[i];
            totalSupport += frame.GetColumn<long>("Support")[i];
        }
        double microAccuracy = (double)totalCorrect / totalSupport;
        Assert.That(ml.MicroAccuracy, Is.EqualTo(microAccuracy).Within(1e-12));
    }

    static float[] OneHot(int index, int length)
    {
        var result = new float[length];
        result[index] = 1f;
        return result;
    }

    private sealed class ClassificationRow
    {
        public float Label { get; set; }
        public float PredictedLabel { get; set; }

        [VectorType(3)]
        public float[] Score { get; set; } = Array.Empty<float>();
    }
}
