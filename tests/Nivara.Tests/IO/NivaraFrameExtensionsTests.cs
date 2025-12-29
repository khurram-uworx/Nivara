using Apache.Arrow;
using Nivara.IO;
using NUnit.Framework;

namespace Nivara.Tests.IO;

[TestFixture]
public class NivaraFrameExtensionsTests
{
    [Test]
    public void ToParquet_WithValidFrame_ShouldNotThrow()
    {
        // Arrange
        var frame = CreateTestFrame();
        var tempFile = Path.GetTempFileName();

        try
        {
            // Act & Assert - should not throw
            Assert.DoesNotThrow(() => frame.ToParquet(tempFile));

            // Verify file was created
            Assert.That(File.Exists(tempFile), Is.True);
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public async Task ToParquetAsync_WithValidFrame_ShouldNotThrow()
    {
        // Arrange
        var frame = CreateTestFrame();
        var tempFile = Path.GetTempFileName();

        try
        {
            // Act & Assert - should not throw
            Assert.DoesNotThrowAsync(async () => await frame.ToParquetAsync(tempFile));

            // Verify file was created
            Assert.That(File.Exists(tempFile), Is.True);
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public void ToParquetStream_WithValidFrame_ShouldNotThrow()
    {
        // Arrange
        var frame = CreateTestFrame();
        using var stream = new MemoryStream();

        // Act & Assert - should not throw
        Assert.DoesNotThrow(() => frame.ToParquetStream(stream));

        // Verify data was written
        Assert.That(stream.Length, Is.GreaterThan(0));
    }

    [Test]
    public async Task ToParquetStreamAsync_WithValidFrame_ShouldNotThrow()
    {
        // Arrange
        var frame = CreateTestFrame();
        using var stream = new MemoryStream();

        // Act & Assert - should not throw
        Assert.DoesNotThrowAsync(async () => await frame.ToParquetStreamAsync(stream));

        // Verify data was written
        Assert.That(stream.Length, Is.GreaterThan(0));
    }

    [Test]
    public void LoadParquet_WithNullPath_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => NivaraFrameExtensions.LoadParquet(null!));
    }

    [Test]
    public void LoadParquetAsync_WithNullPath_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.ThrowsAsync<ArgumentNullException>(async () => await NivaraFrameExtensions.LoadParquetAsync(null!));
    }

    [Test]
    public void LoadParquetFromStream_WithNullStream_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => NivaraFrameExtensions.LoadParquetFromStream(null!));
    }

    [Test]
    public void LoadParquetFromStreamAsync_WithNullStream_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.ThrowsAsync<ArgumentNullException>(async () => await NivaraFrameExtensions.LoadParquetFromStreamAsync(null!));
    }

    [Test]
    public void ToArrowTable_WithValidFrame_ShouldNotThrow()
    {
        // Arrange
        var frame = CreateTestFrame();

        // Act & Assert - should not throw
        Assert.DoesNotThrow(() => frame.ToArrowTable());
    }

    [Test]
    public void ToArrowTable_WithNullFrame_ShouldThrowArgumentNullException()
    {
        // Arrange
        NivaraFrame frame = null!;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => frame.ToArrowTable());
    }

    [Test]
    public void FromArrowTable_WithNullTable_ShouldThrowArgumentNullException()
    {
        // Arrange
        Table table = null!;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => table.FromArrowTable());
    }

    [Test]
    public void ExtensionMethods_ShouldSupportMethodChaining()
    {
        // Arrange
        var frame = CreateTestFrame();

        // Act & Assert - method chaining should work
        Assert.DoesNotThrow(() =>
        {
            var arrowTable = frame.ToArrowTable();
            var roundTripFrame = arrowTable.FromArrowTable();

            // Verify we can chain operations
            Assert.That(roundTripFrame, Is.Not.Null);
            Assert.That(roundTripFrame.ColumnCount, Is.EqualTo(frame.ColumnCount));
        });
    }

    [Test]
    public void ExtensionMethods_ShouldProvideAsyncVariants()
    {
        // Arrange
        var frame = CreateTestFrame();
        var tempFile = Path.GetTempFileName();

        try
        {
            // Act & Assert - both sync and async variants should be available
            Assert.DoesNotThrow(() => frame.ToParquet(tempFile));
            Assert.DoesNotThrowAsync(async () => await frame.ToParquetAsync(tempFile));

            Assert.DoesNotThrow(() => NivaraFrameExtensions.LoadParquet(tempFile));
            Assert.DoesNotThrowAsync(async () => await NivaraFrameExtensions.LoadParquetAsync(tempFile));
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public void ExtensionMethods_ShouldProvideDefaultParameterOverloads()
    {
        // Arrange
        var frame = CreateTestFrame();
        var tempFile = Path.GetTempFileName();

        try
        {
            // Act & Assert - methods should work without options parameter
            Assert.DoesNotThrow(() => frame.ToParquet(tempFile));
            Assert.DoesNotThrow(() => frame.ToArrowTable());

            // And with options parameter
            var parquetOptions = new ParquetWriteOptions();
            var arrowOptions = new ArrowConversionOptions();

            Assert.DoesNotThrow(() => frame.ToParquet(tempFile, parquetOptions));
            Assert.DoesNotThrow(() => frame.ToArrowTable(arrowOptions));
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    private static NivaraFrame CreateTestFrame()
    {
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var stringColumn = NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "c" });

        return NivaraFrame.Create(
            ("IntColumn", intColumn),
            ("StringColumn", stringColumn)
        );
    }
}