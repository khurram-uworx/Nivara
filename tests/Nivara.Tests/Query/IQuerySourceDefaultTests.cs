using Nivara.Helpers;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests.Query;

[TestFixture]
public class IQuerySourceDefaultTests
{
    [Test]
    public void DefaultExecuteAsync_RunsSynchronouslyAndObservesToken()
    {
        var columns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
        var source = new MemoryQuerySource(columns, new Schema(Array.Empty<(string, Type)>()));

        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.Throws<OperationCanceledException>(() =>
            ((IQuerySource)source).ExecuteAsync(cts.Token).GetAwaiter().GetResult());

        cts.Dispose();
    }
}