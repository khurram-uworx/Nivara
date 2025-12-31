using System;
using System.Reflection;

namespace Nivara.Tensors;

/// <summary>
/// Reflection-based factory for constructing TensorStorage instances.
/// Accepts element Type and typed Array to avoid generic constraints on callers.
/// </summary>
internal static class TensorStorageFactory
{
    /// <summary>
    /// Try to create a TensorStorage&lt;T&gt; instance using the provided element type and data array.
    /// Returns null when construction fails.
    /// </summary>
    public static object? TryCreateFromArray(Type elementType, Array dataArray, bool[]? nullMask = null)
    {
        if (elementType == null) throw new ArgumentNullException(nameof(elementType));
        if (dataArray == null) throw new ArgumentNullException(nameof(dataArray));

        try
        {
            var tensorType = typeof(TensorStorage<>).MakeGenericType(elementType);

            // Prefer ctor (T[] data, bool[]? nullMask) then fallback to (T[] data)
            var ctor = tensorType.GetConstructor(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public, null, new Type[] { dataArray.GetType(), typeof(bool[]) }, null)
                       ?? tensorType.GetConstructor(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public, null, new Type[] { dataArray.GetType() }, null)
                       // Fallback when exact runtime array type differs (e.g., elementType[])
                       ?? tensorType.GetConstructor(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public, null, new Type[] { elementType.MakeArrayType(), typeof(bool[]) }, null)
                       ?? tensorType.GetConstructor(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public, null, new Type[] { elementType.MakeArrayType() }, null);

            if (ctor == null)
                return null;

            object? instance;
            var parameters = ctor.GetParameters();
            if (parameters.Length == 2)
                instance = ctor.Invoke(new object?[] { dataArray, nullMask });
            else
                instance = ctor.Invoke(new object?[] { dataArray });

            return instance;
        }
        catch
        {
            return null;
        }
    }
}
