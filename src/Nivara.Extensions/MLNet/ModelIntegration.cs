using Microsoft.ML;
using Microsoft.ML.Data;

namespace Nivara.MLNet;

/// <summary>
/// Provides high-level integration between Nivara DataFrames and ML.NET models.
/// </summary>
public class ModelIntegration
{
    private readonly MLContext mlContext;

    /// <summary>
    /// Initializes a new instance of the ModelIntegration class.
    /// </summary>
    /// <param name="mlContext">The ML.NET context to use</param>
    public ModelIntegration(MLContext mlContext)
    {
        this.mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
    }

    /// <summary>
    /// Creates a binary classification pipeline for NivaraFrame data.
    /// </summary>
    /// <param name="featureColumns">The columns to use as features</param>
    /// <param name="labelColumn">The column containing binary labels</param>
    /// <returns>An ML.NET pipeline configured for binary classification</returns>
    public IEstimator<ITransformer> CreateBinaryClassificationPipeline(
        string[] featureColumns, 
        string labelColumn = "Label")
    {
        if (featureColumns == null || featureColumns.Length == 0)
            throw new ArgumentException("Feature columns cannot be null or empty", nameof(featureColumns));

        return mlContext.Transforms.Concatenate("Features", featureColumns)
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: labelColumn, 
                featureColumnName: "Features"));
    }

    /// <summary>
    /// Creates a multiclass classification pipeline for NivaraFrame data.
    /// </summary>
    /// <param name="featureColumns">The columns to use as features</param>
    /// <param name="labelColumn">The column containing class labels</param>
    /// <returns>An ML.NET pipeline configured for multiclass classification</returns>
    public IEstimator<ITransformer> CreateMulticlassClassificationPipeline(
        string[] featureColumns, 
        string labelColumn = "Label")
    {
        if (featureColumns == null || featureColumns.Length == 0)
            throw new ArgumentException("Feature columns cannot be null or empty", nameof(featureColumns));

        return mlContext.Transforms.Conversion.MapValueToKey(labelColumn)
            .Append(mlContext.Transforms.Concatenate("Features", featureColumns))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                labelColumnName: labelColumn, 
                featureColumnName: "Features"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
    }

    /// <summary>
    /// Creates a regression pipeline for NivaraFrame data.
    /// </summary>
    /// <param name="featureColumns">The columns to use as features</param>
    /// <param name="labelColumn">The column containing target values</param>
    /// <returns>An ML.NET pipeline configured for regression</returns>
    public IEstimator<ITransformer> CreateRegressionPipeline(
        string[] featureColumns, 
        string labelColumn = "Label")
    {
        if (featureColumns == null || featureColumns.Length == 0)
            throw new ArgumentException("Feature columns cannot be null or empty", nameof(featureColumns));

        return mlContext.Transforms.Concatenate("Features", featureColumns)
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.Regression.Trainers.Sdca(
                labelColumnName: labelColumn, 
                featureColumnName: "Features"));
    }

    /// <summary>
    /// Trains a model using a NivaraFrame and evaluates it.
    /// </summary>
    /// <param name="pipeline">The ML.NET pipeline</param>
    /// <param name="trainingData">The training data as a NivaraFrame</param>
    /// <param name="testingData">The testing data as a NivaraFrame</param>
    /// <returns>A trained model and evaluation metrics</returns>
    public (ITransformer Model, object Metrics) TrainAndEvaluate(
        IEstimator<ITransformer> pipeline,
        NivaraFrame trainingData,
        NivaraFrame testingData)
    {
        if (pipeline == null) throw new ArgumentNullException(nameof(pipeline));
        if (trainingData == null) throw new ArgumentNullException(nameof(trainingData));
        if (testingData == null) throw new ArgumentNullException(nameof(testingData));

        // Convert to ML.NET format
        var trainDataView = trainingData.ToDataView(mlContext);
        var testDataView = testingData.ToDataView(mlContext);

        // Train the model
        var model = pipeline.Fit(trainDataView);

        // Make predictions on test data
        var predictions = model.Transform(testDataView);

        // Evaluate based on the type of problem (inferred from pipeline)
        object metrics = EvaluateModel(predictions, pipeline);

        return (model, metrics);
    }

    /// <summary>
    /// Performs cross-validation on a NivaraFrame dataset.
    /// </summary>
    /// <param name="pipeline">The ML.NET pipeline</param>
    /// <param name="data">The data as a NivaraFrame</param>
    /// <param name="numberOfFolds">The number of folds for cross-validation</param>
    /// <returns>Cross-validation results</returns>
    public IEnumerable<TrainCatalogBase.CrossValidationResult<ITransformer>> CrossValidate(
        IEstimator<ITransformer> pipeline,
        NivaraFrame data,
        int numberOfFolds = 5)
    {
        if (pipeline == null) throw new ArgumentNullException(nameof(pipeline));
        if (data == null) throw new ArgumentNullException(nameof(data));
        if (numberOfFolds < 2) throw new ArgumentException("Number of folds must be at least 2", nameof(numberOfFolds));

        var dataView = data.ToDataView(mlContext);

        // Determine the type of cross-validation based on the pipeline
        if (IsBinaryClassificationPipeline(pipeline))
        {
            return mlContext.BinaryClassification.CrossValidate(dataView, pipeline, numberOfFolds)
                .Cast<TrainCatalogBase.CrossValidationResult<ITransformer>>();
        }
        else if (IsMulticlassClassificationPipeline(pipeline))
        {
            return mlContext.MulticlassClassification.CrossValidate(dataView, pipeline, numberOfFolds)
                .Cast<TrainCatalogBase.CrossValidationResult<ITransformer>>();
        }
        else if (IsRegressionPipeline(pipeline))
        {
            return mlContext.Regression.CrossValidate(dataView, pipeline, numberOfFolds)
                .Cast<TrainCatalogBase.CrossValidationResult<ITransformer>>();
        }
        else
        {
            throw new NotSupportedException("Pipeline type not supported for cross-validation");
        }
    }

    /// <summary>
    /// Creates feature importance analysis for a trained model.
    /// </summary>
    /// <param name="model">The trained ML.NET model</param>
    /// <param name="data">The data used for analysis</param>
    /// <param name="featureColumns">The feature column names</param>
    /// <returns>Feature importance scores</returns>
    public Dictionary<string, float> AnalyzeFeatureImportance(
        ITransformer model,
        NivaraFrame data,
        string[] featureColumns)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));
        if (data == null) throw new ArgumentNullException(nameof(data));
        if (featureColumns == null) throw new ArgumentNullException(nameof(featureColumns));

        var dataView = data.ToDataView(mlContext);
        var transformedData = model.Transform(dataView);

        // Use ML.NET's feature importance analysis
        var featureImportance = new Dictionary<string, float>();

        try
        {
            // Try to get feature importance from the model if it supports it
            // For most models, we'll use permutation feature importance
            var permutationMetrics = mlContext.Regression.PermutationFeatureImportance(
                model, transformedData, permutationCount: 10);

            for (int i = 0; i < Math.Min(permutationMetrics.Count, featureColumns.Length); i++)
            {
                featureImportance[featureColumns[i]] = (float)Math.Abs(permutationMetrics[featureColumns[i]].RSquared.Mean);
            }
        }
        catch
        {
            // If feature importance analysis fails, return empty dictionary
            // This can happen with some model types that don't support it
        }

        return featureImportance;
    }

    /// <summary>
    /// Saves a trained model to disk.
    /// </summary>
    /// <param name="model">The trained model</param>
    /// <param name="trainingData">The training data schema</param>
    /// <param name="filePath">The path to save the model</param>
    public void SaveModel(ITransformer model, NivaraFrame trainingData, string filePath)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));
        if (trainingData == null) throw new ArgumentNullException(nameof(trainingData));
        if (string.IsNullOrEmpty(filePath)) throw new ArgumentException("File path cannot be null or empty", nameof(filePath));

        var dataView = trainingData.ToDataView(mlContext);
        mlContext.Model.Save(model, dataView.Schema, filePath);
    }

    /// <summary>
    /// Loads a trained model from disk.
    /// </summary>
    /// <param name="filePath">The path to the saved model</param>
    /// <returns>The loaded model and its input schema</returns>
    public (ITransformer Model, DataViewSchema Schema) LoadModel(string filePath)
    {
        if (string.IsNullOrEmpty(filePath)) throw new ArgumentException("File path cannot be null or empty", nameof(filePath));

        using var stream = File.OpenRead(filePath);
        var model = mlContext.Model.Load(stream, out var schema);
        return (model, schema);
    }

    // Private helper methods
    private object EvaluateModel(IDataView predictions, IEstimator<ITransformer> pipeline)
    {
        if (IsBinaryClassificationPipeline(pipeline))
        {
            return mlContext.BinaryClassification.Evaluate(predictions);
        }
        else if (IsMulticlassClassificationPipeline(pipeline))
        {
            return mlContext.MulticlassClassification.Evaluate(predictions);
        }
        else if (IsRegressionPipeline(pipeline))
        {
            return mlContext.Regression.Evaluate(predictions);
        }
        else
        {
            throw new NotSupportedException("Pipeline type not supported for evaluation");
        }
    }

    private bool IsBinaryClassificationPipeline(IEstimator<ITransformer> pipeline)
    {
        // Simple heuristic - check if pipeline contains binary classification trainer
        return pipeline.ToString()?.Contains("BinaryClassification") == true;
    }

    private bool IsMulticlassClassificationPipeline(IEstimator<ITransformer> pipeline)
    {
        // Simple heuristic - check if pipeline contains multiclass classification trainer
        return pipeline.ToString()?.Contains("MulticlassClassification") == true;
    }

    private bool IsRegressionPipeline(IEstimator<ITransformer> pipeline)
    {
        // Simple heuristic - check if pipeline contains regression trainer
        return pipeline.ToString()?.Contains("Regression") == true;
    }
}