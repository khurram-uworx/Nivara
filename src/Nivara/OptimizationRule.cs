namespace Nivara;

/// <summary>
/// Abstract base class for query optimization rules
/// </summary>
public abstract class OptimizationRule
{
    /// <summary>
    /// Gets the name of this optimization rule
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Gets the description of what this optimization rule does
    /// </summary>
    public abstract string Description { get; }

    /// <summary>
    /// Gets the priority of this rule (higher values are applied first)
    /// </summary>
    public virtual int Priority => 0;

    /// <summary>
    /// Determines if this rule can be applied to the given query plan
    /// </summary>
    /// <param name="plan">The query plan to check</param>
    /// <returns>True if the rule can be applied, false otherwise</returns>
    public abstract bool CanApply(QueryPlan plan);

    /// <summary>
    /// Applies this optimization rule to the query plan
    /// </summary>
    /// <param name="plan">The query plan to optimize</param>
    /// <returns>The optimized query plan</returns>
    public abstract QueryPlan Apply(QueryPlan plan);

    /// <summary>
    /// Gets statistics about the optimization applied
    /// </summary>
    /// <param name="originalPlan">The original query plan</param>
    /// <param name="optimizedPlan">The optimized query plan</param>
    /// <returns>Statistics about the optimization</returns>
    public virtual OptimizationStatistics GetStatistics(QueryPlan originalPlan, QueryPlan optimizedPlan)
    {
        return new OptimizationStatistics
        {
            RuleName = Name,
            OriginalOperationCount = originalPlan.Operations.Count,
            OptimizedOperationCount = optimizedPlan.Operations.Count,
            EstimatedImprovementPercent = CalculateEstimatedImprovement(originalPlan, optimizedPlan)
        };
    }

    /// <summary>
    /// Calculates the estimated performance improvement percentage
    /// </summary>
    /// <param name="originalPlan">The original query plan</param>
    /// <param name="optimizedPlan">The optimized query plan</param>
    /// <returns>Estimated improvement as a percentage</returns>
    protected virtual double CalculateEstimatedImprovement(QueryPlan originalPlan, QueryPlan optimizedPlan)
    {
        // Simple heuristic: fewer operations generally means better performance
        if (originalPlan.Operations.Count == 0)
            return 0.0;

        var operationReduction = originalPlan.Operations.Count - optimizedPlan.Operations.Count;
        return (operationReduction / (double)originalPlan.Operations.Count) * 100.0;
    }

    /// <summary>
    /// Returns a string representation of this optimization rule
    /// </summary>
    /// <returns>A formatted string describing the rule</returns>
    public override string ToString()
    {
        return $"{Name} (Priority: {Priority})";
    }
}

/// <summary>
/// Statistics about an applied optimization rule
/// </summary>
public sealed class OptimizationStatistics
{
    /// <summary>
    /// Gets or sets the name of the optimization rule
    /// </summary>
    public string RuleName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the number of operations in the original plan
    /// </summary>
    public int OriginalOperationCount { get; set; }

    /// <summary>
    /// Gets or sets the number of operations in the optimized plan
    /// </summary>
    public int OptimizedOperationCount { get; set; }

    /// <summary>
    /// Gets or sets the estimated performance improvement percentage
    /// </summary>
    public double EstimatedImprovementPercent { get; set; }

    /// <summary>
    /// Gets or sets the time taken to apply the optimization
    /// </summary>
    public TimeSpan OptimizationTime { get; set; }

    /// <summary>
    /// Gets or sets additional metadata about the optimization
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; } = new();

    /// <summary>
    /// Returns a string representation of these statistics
    /// </summary>
    /// <returns>A formatted string describing the statistics</returns>
    public override string ToString()
    {
        return $"{RuleName}: {OriginalOperationCount} -> {OptimizedOperationCount} operations " +
               $"({EstimatedImprovementPercent:F1}% improvement, {OptimizationTime.TotalMilliseconds:F2}ms)";
    }
}

/// <summary>
/// Engine for applying optimization rules to query plans
/// </summary>
public sealed class OptimizationEngine
{
    private readonly List<OptimizationRule> _rules;

    /// <summary>
    /// Initializes a new instance of OptimizationEngine
    /// </summary>
    public OptimizationEngine()
    {
        _rules = new List<OptimizationRule>();
    }

    /// <summary>
    /// Initializes a new instance of OptimizationEngine with the specified rules
    /// </summary>
    /// <param name="rules">The optimization rules to use</param>
    public OptimizationEngine(IEnumerable<OptimizationRule> rules)
    {
        _rules = rules?.ToList() ?? throw new ArgumentNullException(nameof(rules));
    }

    /// <summary>
    /// Gets the optimization rules registered with this engine
    /// </summary>
    public IReadOnlyList<OptimizationRule> Rules => _rules;

    /// <summary>
    /// Adds an optimization rule to the engine
    /// </summary>
    /// <param name="rule">The rule to add</param>
    public void AddRule(OptimizationRule rule)
    {
        if (rule == null)
            throw new ArgumentNullException(nameof(rule));

        _rules.Add(rule);
        _rules.Sort((a, b) => b.Priority.CompareTo(a.Priority)); // Sort by priority descending
    }

    /// <summary>
    /// Removes an optimization rule from the engine
    /// </summary>
    /// <param name="rule">The rule to remove</param>
    /// <returns>True if the rule was removed, false if it wasn't found</returns>
    public bool RemoveRule(OptimizationRule rule)
    {
        return _rules.Remove(rule);
    }

    /// <summary>
    /// Optimizes a query plan by applying all applicable rules
    /// </summary>
    /// <param name="plan">The query plan to optimize</param>
    /// <returns>The optimization result containing the optimized plan and statistics</returns>
    public OptimizationResult Optimize(QueryPlan plan)
    {
        if (plan == null)
            throw new ArgumentNullException(nameof(plan));

        var result = new OptimizationResult
        {
            OriginalPlan = plan,
            OptimizedPlan = plan,
            Statistics = new List<OptimizationStatistics>()
        };

        var currentPlan = plan;
        var totalOptimizationTime = TimeSpan.Zero;

        foreach (var rule in _rules)
        {
            if (rule.CanApply(currentPlan))
            {
                var startTime = DateTime.UtcNow;

                try
                {
                    var optimizedPlan = rule.Apply(currentPlan);
                    var optimizationTime = DateTime.UtcNow - startTime;
                    totalOptimizationTime += optimizationTime;

                    var statistics = rule.GetStatistics(currentPlan, optimizedPlan);
                    statistics.OptimizationTime = optimizationTime;

                    result.Statistics.Add(statistics);
                    currentPlan = optimizedPlan;
                }
                catch (Exception ex)
                {
                    // If an optimization rule fails, skip it and continue with other rules
                    // This ensures that optimization failures don't break the entire query
                    var failureStats = new OptimizationStatistics
                    {
                        RuleName = rule.Name,
                        OriginalOperationCount = currentPlan.Operations.Count,
                        OptimizedOperationCount = currentPlan.Operations.Count,
                        EstimatedImprovementPercent = 0.0,
                        OptimizationTime = DateTime.UtcNow - startTime,
                        Metadata = { ["Error"] = ex.Message }
                    };
                    result.Statistics.Add(failureStats);
                }
            }
        }

        result.OptimizedPlan = currentPlan;
        result.TotalOptimizationTime = totalOptimizationTime;

        return result;
    }

    /// <summary>
    /// Creates a default optimization engine with standard rules
    /// </summary>
    /// <returns>An optimization engine with standard optimization rules</returns>
    public static OptimizationEngine CreateDefault()
    {
        var engine = new OptimizationEngine();

        // Add standard optimization rules in priority order
        engine.AddRule(new PredicatePushdownRule());
        engine.AddRule(new ProjectionPushdownRule());
        engine.AddRule(new OperationFusionRule());
        engine.AddRule(new ColumnEliminationRule());

        return engine;
    }
}

/// <summary>
/// Result of applying optimization rules to a query plan
/// </summary>
public sealed class OptimizationResult
{
    /// <summary>
    /// Gets or sets the original query plan
    /// </summary>
    public QueryPlan OriginalPlan { get; set; } = null!;

    /// <summary>
    /// Gets or sets the optimized query plan
    /// </summary>
    public QueryPlan OptimizedPlan { get; set; } = null!;

    /// <summary>
    /// Gets or sets the statistics for each applied optimization
    /// </summary>
    public List<OptimizationStatistics> Statistics { get; set; } = new();

    /// <summary>
    /// Gets or sets the total time spent on optimization
    /// </summary>
    public TimeSpan TotalOptimizationTime { get; set; }

    /// <summary>
    /// Gets the total estimated improvement percentage
    /// </summary>
    public double TotalEstimatedImprovement => Statistics.Sum(s => s.EstimatedImprovementPercent);

    /// <summary>
    /// Gets the number of rules that were successfully applied
    /// </summary>
    public int AppliedRuleCount => Statistics.Count(s => !s.Metadata.ContainsKey("Error"));

    /// <summary>
    /// Gets the number of rules that failed to apply
    /// </summary>
    public int FailedRuleCount => Statistics.Count(s => s.Metadata.ContainsKey("Error"));

    /// <summary>
    /// Generates a detailed report of the optimization results
    /// </summary>
    /// <returns>A formatted report string</returns>
    public string GenerateReport()
    {
        var report = new System.Text.StringBuilder();

        report.AppendLine("Query Optimization Report");
        report.AppendLine("========================");
        report.AppendLine();

        report.AppendLine($"Original Operations: {OriginalPlan.Operations.Count}");
        report.AppendLine($"Optimized Operations: {OptimizedPlan.Operations.Count}");
        report.AppendLine($"Total Optimization Time: {TotalOptimizationTime.TotalMilliseconds:F2}ms");
        report.AppendLine($"Applied Rules: {AppliedRuleCount}");
        report.AppendLine($"Failed Rules: {FailedRuleCount}");
        report.AppendLine($"Estimated Improvement: {TotalEstimatedImprovement:F1}%");
        report.AppendLine();

        if (Statistics.Count > 0)
        {
            report.AppendLine("Rule Details:");
            foreach (var stat in Statistics)
            {
                report.AppendLine($"  {stat}");
                if (stat.Metadata.ContainsKey("Error"))
                {
                    report.AppendLine($"    Error: {stat.Metadata["Error"]}");
                }
            }
            report.AppendLine();
        }

        report.AppendLine("Original Plan:");
        report.AppendLine(QueryPlanAnalyzer.Explain(OriginalPlan));
        report.AppendLine();

        report.AppendLine("Optimized Plan:");
        report.AppendLine(QueryPlanAnalyzer.Explain(OptimizedPlan));

        return report.ToString();
    }
}