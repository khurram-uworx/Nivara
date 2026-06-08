namespace Nivara.Query;

/// <summary>
/// Defines the diagnostic modes available for query analysis and debugging.
/// Controls the level of detail and type of diagnostic information provided.
/// </summary>
public enum QueryDiagnosticMode
{
    /// <summary>
    /// No diagnostic information is collected or provided
    /// </summary>
    None,

    /// <summary>
    /// Basic query plan structure and operation sequence
    /// </summary>
    Basic,

    /// <summary>
    /// Detailed query plan with schema transformations and operation details
    /// </summary>
    Detailed,

    /// <summary>
    /// Performance analysis including optimization opportunities and execution estimates
    /// </summary>
    Performance,

    /// <summary>
    /// Comprehensive diagnostic information including all available details,
    /// optimization analysis, performance estimates, and debugging information
    /// </summary>
    Comprehensive
}

/// <summary>
/// Provides diagnostic information and analysis for queries.
/// Supports different levels of detail based on the diagnostic mode.
/// </summary>
public static class QueryDiagnostics
{
    /// <summary>
    /// Gets or sets the global diagnostic mode for all queries
    /// </summary>
    public static QueryDiagnosticMode GlobalMode { get; set; } = QueryDiagnosticMode.None;

    /// <summary>
    /// Gets diagnostic information for a query plan based on the specified mode
    /// </summary>
    /// <param name="queryPlan">The query plan to analyze</param>
    /// <param name="mode">The diagnostic mode to use</param>
    /// <returns>Diagnostic information formatted according to the mode</returns>
    public static string GetDiagnosticInfo(QueryPlan queryPlan, QueryDiagnosticMode mode)
    {
        if (queryPlan == null)
            throw new ArgumentNullException(nameof(queryPlan));

        return mode switch
        {
            QueryDiagnosticMode.None => string.Empty,
            QueryDiagnosticMode.Basic => GetBasicDiagnostics(queryPlan),
            QueryDiagnosticMode.Detailed => GetDetailedDiagnostics(queryPlan),
            QueryDiagnosticMode.Performance => GetPerformanceDiagnostics(queryPlan),
            QueryDiagnosticMode.Comprehensive => GetComprehensiveDiagnostics(queryPlan),
            _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, "Unknown diagnostic mode")
        };
    }

    /// <summary>
    /// Gets diagnostic information using the global diagnostic mode
    /// </summary>
    /// <param name="queryPlan">The query plan to analyze</param>
    /// <returns>Diagnostic information formatted according to the global mode</returns>
    public static string GetDiagnosticInfo(QueryPlan queryPlan)
    {
        return GetDiagnosticInfo(queryPlan, GlobalMode);
    }

    /// <summary>
    /// Analyzes a query plan for potential issues and provides recommendations
    /// </summary>
    /// <param name="queryPlan">The query plan to analyze</param>
    /// <returns>A list of diagnostic recommendations</returns>
    public static IReadOnlyList<string> AnalyzeQueryPlan(QueryPlan queryPlan)
    {
        if (queryPlan == null)
            throw new ArgumentNullException(nameof(queryPlan));

        var recommendations = new List<string>();

        // Check for common performance issues
        AnalyzePerformanceIssues(queryPlan, recommendations);

        // Check for potential correctness issues
        AnalyzeCorrectnessIssues(queryPlan, recommendations);

        // Check for resource usage issues
        AnalyzeResourceUsage(queryPlan, recommendations);

        return recommendations;
    }

    private static string GetBasicDiagnostics(QueryPlan queryPlan)
    {
        var info = new System.Text.StringBuilder();
        info.AppendLine($"Query Plan: {queryPlan.Operations.Count} operations");
        info.AppendLine($"Source: {queryPlan.Source.GetType().Name} ({(queryPlan.Source.IsLazy ? "Lazy" : "Eager")})");
        info.AppendLine($"Input Columns: {queryPlan.Source.Schema.ColumnNames.Count}");
        info.AppendLine($"Output Columns: {queryPlan.ResultSchema.ColumnNames.Count}");

        if (queryPlan.Operations.Count > 0)
        {
            var operationTypes = queryPlan.Operations.Select(op => op.OperationType);
            info.AppendLine($"Operations: {string.Join(" → ", operationTypes)}");
        }

        return info.ToString();
    }

    private static string GetDetailedDiagnostics(QueryPlan queryPlan)
    {
        return QueryPlanAnalyzer.Explain(queryPlan);
    }

    private static string GetPerformanceDiagnostics(QueryPlan queryPlan)
    {
        var info = new System.Text.StringBuilder();
        info.AppendLine("Performance Analysis:");
        info.AppendLine("===================");

        // Execution cost estimate
        var executor = new QueryExecutor();
        var estimatedCost = executor.EstimateExecutionCost(queryPlan);
        info.AppendLine($"Estimated Execution Cost: {estimatedCost}");

        // Optimization opportunities
        var optimizations = QueryPlanAnalyzer.AnalyzeOptimizations(queryPlan);
        if (optimizations.Count > 0)
        {
            info.AppendLine("\nOptimization Opportunities:");
            foreach (var optimization in optimizations)
            {
                info.AppendLine($"  • {optimization}");
            }
        }
        else
        {
            info.AppendLine("\nNo optimization opportunities identified.");
        }

        // Resource usage analysis
        info.AppendLine($"\nResource Analysis:");
        info.AppendLine($"  Data Source Type: {(queryPlan.Source.IsLazy ? "Lazy (deferred I/O)" : "Eager (immediate I/O)")}");
        info.AppendLine($"  Pipeline Length: {queryPlan.Operations.Count} operations");
        info.AppendLine($"  Column Reduction: {queryPlan.Source.Schema.ColumnNames.Count} → {queryPlan.ResultSchema.ColumnNames.Count}");

        return info.ToString();
    }

    private static string GetComprehensiveDiagnostics(QueryPlan queryPlan)
    {
        var info = new System.Text.StringBuilder();

        // Include detailed diagnostics
        info.AppendLine(GetDetailedDiagnostics(queryPlan));
        info.AppendLine();

        // Include performance diagnostics
        info.AppendLine(GetPerformanceDiagnostics(queryPlan));
        info.AppendLine();

        // Include recommendations
        var recommendations = AnalyzeQueryPlan(queryPlan);
        if (recommendations.Count > 0)
        {
            info.AppendLine("Recommendations:");
            info.AppendLine("===============");
            foreach (var recommendation in recommendations)
            {
                info.AppendLine($"  • {recommendation}");
            }
        }

        return info.ToString();
    }

    private static void AnalyzePerformanceIssues(QueryPlan queryPlan, List<string> recommendations)
    {
        // Check for expensive operations without filtering
        var hasGroupBy = queryPlan.Operations.Any(op => op.OperationType == OperationType.GroupBy);
        var hasFilter = queryPlan.Operations.Any(op => op.OperationType == OperationType.Filter);

        if (hasGroupBy && !hasFilter)
        {
            recommendations.Add("GroupBy operation without filtering may process unnecessary data - consider adding filters");
        }

        // Check for multiple select operations
        var selectCount = queryPlan.Operations.Count(op => op.OperationType == OperationType.Select);
        if (selectCount > 1)
        {
            recommendations.Add($"Multiple Select operations ({selectCount}) detected - consider combining into single projection");
        }

        // Check for operation ordering
        var operationTypes = queryPlan.Operations.Select(op => op.OperationType).ToList();
        for (int i = 0; i < operationTypes.Count - 1; i++)
        {
            if (operationTypes[i] == OperationType.Select && operationTypes[i + 1] == OperationType.Filter)
            {
                recommendations.Add("Filter operation after Select - consider reordering for better performance");
            }
        }
    }

    private static void AnalyzeCorrectnessIssues(QueryPlan queryPlan, List<string> recommendations)
    {
        // Check for schema compatibility issues
        try
        {
            var currentSchema = queryPlan.Source.Schema;
            foreach (var operation in queryPlan.Operations)
            {
                currentSchema = operation.TransformSchema(currentSchema);
            }
        }
        catch (Exception ex)
        {
            recommendations.Add($"Schema transformation error detected: {ex.Message}");
        }

        // Check for empty result scenarios
        if (queryPlan.ResultSchema.ColumnNames.Count == 0)
        {
            recommendations.Add("Query results in no columns - verify Select operations");
        }
    }

    private static void AnalyzeResourceUsage(QueryPlan queryPlan, List<string> recommendations)
    {
        // Check for lazy source with many operations
        if (queryPlan.Source.IsLazy && queryPlan.Operations.Count > 5)
        {
            recommendations.Add("Complex query on lazy source - consider materializing intermediate results for better performance");
        }

        // Check for column elimination opportunities
        var inputColumns = queryPlan.Source.Schema.ColumnNames.Count;
        var outputColumns = queryPlan.ResultSchema.ColumnNames.Count;
        var reductionRatio = (double)outputColumns / inputColumns;

        if (reductionRatio < 0.5)
        {
            recommendations.Add($"Significant column reduction ({inputColumns} → {outputColumns}) - ensure early column elimination for memory efficiency");
        }
    }
}