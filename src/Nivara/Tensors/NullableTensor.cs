using System.Numerics.Tensors;

namespace Nivara.Tensors;

/// <summary>
/// Tensor data plus an optional explicit null mask.
/// </summary>
/// <typeparam name="T">The tensor element type.</typeparam>
/// <param name="Data">Tensor data. Values at null positions are ignored when <paramref name="NullMask"/> is present.</param>
/// <param name="NullMask">Optional mask where true means the corresponding data position is null.</param>
public sealed record NullableTensor<T>(Tensor<T> Data, Tensor<bool>? NullMask);
