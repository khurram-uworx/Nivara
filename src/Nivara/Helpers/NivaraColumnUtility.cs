using System.Numerics;

namespace Nivara.Helpers;

class NivaraColumnUtility
{
    public static bool MergeNullMasks<T>(NivaraColumn<T> a, NivaraColumn<T> b, Span<bool> destination) where T : struct, INumber<T>
    {
        var aHasNulls = a.TryGetNullMask(out var aMask);
        var bHasNulls = b.TryGetNullMask(out var bMask);

        if (aHasNulls && bHasNulls)
            for (int i = 0; i < destination.Length; i++)
                destination[i] = aMask[i] || bMask[i];
        else if (aHasNulls)
            aMask.CopyTo(destination);
        else if (bHasNulls)
            bMask.CopyTo(destination);

        return aHasNulls || bHasNulls;
    }

}
