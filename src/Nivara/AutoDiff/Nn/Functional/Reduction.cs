namespace Nivara.AutoDiff.Nn.Functional;

/// <summary>Selects how a loss is reduced to a scalar (or kept element-wise).</summary>
public enum Reduction
{
    /// <summary>The loss elements are summed.</summary>
    Sum,
    /// <summary>The loss elements are averaged.</summary>
    Mean,
    /// <summary>The element-wise loss is returned without reduction.</summary>
    None
}
