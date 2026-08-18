namespace Nivara.Operations;

/// <summary>
/// Multi-column comparer that pre-captures all column references once via
/// <c>Func&lt;int, int, int&gt;</c> delegates, eliminating per-comparison
/// dictionary lookup and type-switch dispatch.
/// Stable sort via original-index tiebreaker.
/// </summary>
internal sealed class PreCapturedMultiColumnComparer : IComparer<int>
{
    readonly Func<int, int, int>[] keyComparers;

    public PreCapturedMultiColumnComparer(Func<int, int, int>[] keyComparers)
    {
        this.keyComparers = keyComparers;
    }

    public int Compare(int x, int y)
    {
        for (int i = 0; i < keyComparers.Length; i++)
        {
            int cmp = keyComparers[i](x, y);
            if (cmp != 0)
                return cmp;
        }
        return x.CompareTo(y);
    }
}

/// <summary>
/// Factory methods that create <see cref="Func{Int32, Int32, Int32}"/> delegates
/// for each column type, capturing the column (or backing array) once at creation time.
/// </summary>
internal static class SortKeyComparerFactory
{
    public static Func<int, int, int>? TryCreateKeyComparer(IColumn column, SortKey sortKey)
    {
        return column switch
        {
            NivaraColumn<bool> c => Create(c, sortKey),
            NivaraColumn<char> c => Create(c, sortKey),
            NivaraColumn<byte> c => Create(c, sortKey),
            NivaraColumn<sbyte> c => Create(c, sortKey),
            NivaraColumn<short> c => Create(c, sortKey),
            NivaraColumn<ushort> c => Create(c, sortKey),
            NivaraColumn<int> c => Create(c, sortKey),
            NivaraColumn<uint> c => Create(c, sortKey),
            NivaraColumn<long> c => Create(c, sortKey),
            NivaraColumn<ulong> c => Create(c, sortKey),
            NivaraColumn<nint> c => Create(c, sortKey),
            NivaraColumn<nuint> c => Create(c, sortKey),
            NivaraColumn<Int128> c => Create(c, sortKey),
            NivaraColumn<UInt128> c => Create(c, sortKey),
            NivaraColumn<float> c => Create(c, sortKey),
            NivaraColumn<double> c => Create(c, sortKey),
            NivaraColumn<Half> c => Create(c, sortKey),
            NivaraColumn<decimal> c => Create(c, sortKey),
            NivaraColumn<string> c => Create(c, sortKey),
            NivaraColumn<Guid> c => Create(c, sortKey),
            NivaraColumn<DateTime> c => Create(c, sortKey),
            NivaraColumn<DateTimeOffset> c => Create(c, sortKey),
            NivaraColumn<TimeSpan> c => Create(c, sortKey),
            NivaraColumn<DateOnly> c => Create(c, sortKey),
            NivaraColumn<TimeOnly> c => Create(c, sortKey),
            _ => null
        };

        static Func<int, int, int> Create<T>(NivaraColumn<T> col, SortKey key) where T : IComparable<T>
        {
            bool desc = key.Direction == SortDirection.Descending;
            NullOrdering nullOrd = key.NullOrdering;

            if (col.TryGetSpan(out _))
            {
                var data = col.ToArray();
                return (x, y) =>
                {
                    int cmp = data[x].CompareTo(data[y]);
                    if (cmp != 0) return desc ? -cmp : cmp;
                    return 0;
                };
            }
            else
            {
                var c = col;
                return (x, y) =>
                {
                    bool xNull = c.IsNull(x);
                    bool yNull = c.IsNull(y);
                    if (xNull && yNull) return 0;
                    if (xNull) return nullOrd == NullOrdering.NullsFirst ? -1 : 1;
                    if (yNull) return nullOrd == NullOrdering.NullsFirst ? 1 : -1;
                    int cmp = c[x].CompareTo(c[y]);
                    if (cmp != 0) return desc ? -cmp : cmp;
                    return 0;
                };
            }
        }
    }

    public static bool TryCreatePreCapturedComparer(
        IReadOnlyDictionary<string, IColumn> input, IReadOnlyList<SortKey> sortKeys, out IComparer<int> comparer)
    {
        comparer = null!;
        var keyComparers = new Func<int, int, int>[sortKeys.Count];

        for (int i = 0; i < sortKeys.Count; i++)
        {
            if (!input.TryGetValue(sortKeys[i].ColumnName, out var column))
                return false;

            var kc = TryCreateKeyComparer(column, sortKeys[i]);
            if (kc == null)
                return false;

            keyComparers[i] = kc;
        }

        comparer = new PreCapturedMultiColumnComparer(keyComparers);
        return true;
    }
}
