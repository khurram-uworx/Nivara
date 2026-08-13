using System.Collections.Generic;

namespace Nivara.AutoDiff.Utilities;

/// <summary>
/// Strongly-typed diagnostic summary of a computation graph rooted at a tensor.
/// Eliminates the boxing inherent in the previous <c>Dictionary&lt;string, object&gt;</c>
/// representation and gives compile-time access to each field.
/// </summary>
public readonly record struct GraphInfo(
    int TotalNodes,
    bool IsLeaf,
    bool RequiresGrad,
    IReadOnlyDictionary<string, int> OperationCounts);
