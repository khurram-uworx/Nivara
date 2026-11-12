using Nivara.Storage;
using NUnit.Framework;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;

namespace Nivara.Tests.Storage;

/// <summary>
/// Concurrency regression tests for the lazy caches in <see cref="ColumnStorage{T}"/>.
/// Categorised as Performance so they do not run in the default (filtered) suite.
/// </summary>
[TestFixture]
[Category("Performance")]
public class ColumnStorageConcurrencyTests
{
    [Test]
    public void ColumnStorage_AsTensor_ConcurrentCallers_ReturnSameInstance()
    {
        var storage = new ColumnStorage<int>(Enumerable.Range(0, 1_000).ToArray());
        var views = new Tensor<int>[8];

        Parallel.For(0, views.Length, i => views[i] = storage.AsTensor());

        for (int i = 1; i < views.Length; i++)
            Assert.That(ReferenceEquals(views[0], views[i]), Is.True,
                $"Concurrent AsTensor() callers must share one cached view (index {i}).");
    }

    [Test]
    public void ColumnStorage_TryGetHashCodes_ConcurrentCallers_SeeIdenticalContents()
    {
        var values = Enumerable.Range(0, 1_000).Select(i => $"v{i}").ToArray();
        var storage = new ColumnStorage<string>(values, detectNulls: true);
        var results = new int[8][];

        Parallel.For(0, results.Length, i =>
        {
            if (storage.TryGetHashCodes(out var span))
                results[i] = span.ToArray();
        });

        for (int i = 1; i < results.Length; i++)
            Assert.That(results[i], Is.EqualTo(results[0]),
                $"Concurrent TryGetHashCodes callers must observe identical contents (index {i}).");
    }
}
