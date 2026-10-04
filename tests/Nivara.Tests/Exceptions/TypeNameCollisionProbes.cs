namespace Nivara.GateProbe.Alpha
{
    /// <summary>
    /// Half of a deliberately colliding pair. See <c>TypeNameCollisionProbes</c>.
    /// </summary>
    internal sealed class DuplicateName;
}

namespace Nivara.GateProbe.Beta
{
    /// <summary>
    /// The other half. Same short name, different namespace -- which is exactly the shape
    /// that produces CS0104 in a consumer importing both.
    /// </summary>
    internal sealed class DuplicateName;
}