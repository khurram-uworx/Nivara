using Nivara.IO;
using NUnit.Framework;

namespace Nivara.Tests.IO;

[TestFixture]
public class ParquetWriterTests
{
    [Test]
    public void WriteParquet_WithValidFrame_CreatesFile()
    {
        // Arrange
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var stringColumn = NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "c", "d", "e" });
        var frame = NivaraFrame.Create(
            ("IntColumn", intColumn),
            ("StringColumn", stringColumn)
        );

        var tempFile = Path.GetTempFileName();
        var options = new ParquetWriteOptions { ValidateSchema = true };

        try
        {
            // Act & Assert - should not throw
            Assert.DoesNotThrow(() => ParquetWriter.WriteParquet(frame, tempFile, options));

            // Verify file was created
            Assert.That(File.Exists(tempFile), Is.True);
            Assert.That(new FileInfo(tempFile).Length, Is.GreaterThan(0));
        }
        finally
        {
            // Cleanup
            frame.Dispose();
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public async Task WriteParquetAsync_WithValidFrame_CreatesFile()
    {
        // Arrange
        var doubleColumn = NivaraColumn<double>.Create(new[] { 1.1, 2.2, 3.3 });
        var boolColumn = NivaraColumn<bool>.Create(new[] { true, false, true });
        var frame = NivaraFrame.Create(
            ("DoubleColumn", doubleColumn),
            ("BoolColumn", boolColumn)
        );

        var tempFile = Path.GetTempFileName();

        try
        {
            // Act & Assert - should not throw
            Assert.DoesNotThrowAsync(async () => await ParquetWriter.WriteParquetAsync(frame, tempFile));

            // Verify file was created
            Assert.That(File.Exists(tempFile), Is.True);
            Assert.That(new FileInfo(tempFile).Length, Is.GreaterThan(0));
        }
        finally
        {
            // Cleanup
            frame.Dispose();
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public void WriteParquet_WithEmptyFrame_CreatesValidFile()
    {
        // Arrange
        var emptyColumn = NivaraColumn<int>.Create(Array.Empty<int>());
        var frame = NivaraFrame.Create(("EmptyColumn", emptyColumn));

        var tempFile = Path.GetTempFileName();

        try
        {
            // Act & Assert - should not throw
            Assert.DoesNotThrow(() => ParquetWriter.WriteParquet(frame, tempFile));

            // Verify file was created
            Assert.That(File.Exists(tempFile), Is.True);
            Assert.That(new FileInfo(tempFile).Length, Is.GreaterThan(0));
        }
        finally
        {
            // Cleanup
            frame.Dispose();
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public void WriteParquet_WithNullFrame_ThrowsArgumentNullException()
    {
        // Arrange
        var tempFile = Path.GetTempFileName();

        try
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => ParquetWriter.WriteParquet(null!, tempFile));
        }
        finally
        {
            // Cleanup
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public void WriteParquet_WithNullFilePath_ThrowsArgumentNullException()
    {
        // Arrange
        var column = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("TestColumn", column));

        try
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => ParquetWriter.WriteParquet(frame, (string)null!));
        }
        finally
        {
            // Cleanup
            frame.Dispose();
        }
    }

    [Test]
    public void WriteParquet_WithStream_WritesSuccessfully()
    {
        // Arrange
        var longColumn = NivaraColumn<long>.Create(new[] { 100L, 200L, 300L });
        var frame = NivaraFrame.Create(("LongColumn", longColumn));

        using var stream = new MemoryStream();
        var options = new ParquetWriteOptions();

        try
        {
            // Act & Assert - should not throw
            Assert.DoesNotThrow(() => ParquetWriter.WriteParquet(frame, stream, options));

            // Verify data was written
            Assert.That(stream.Length, Is.GreaterThan(0));
        }
        finally
        {
            // Cleanup
            frame.Dispose();
        }
    }

    [Test]
    public void WriteParquetBatch_WithMultipleFrames_WritesSuccessfully()
    {
        // Arrange
        var frame1 = NivaraFrame.Create(
            ("IntColumn", NivaraColumn<int>.Create(new[] { 1, 2 })),
            ("StringColumn", NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b" }))
        );

        var frame2 = NivaraFrame.Create(
            ("IntColumn", NivaraColumn<int>.Create(new[] { 3, 4 })),
            ("StringColumn", NivaraColumn<string>.CreateForReferenceType(new[] { "c", "d" }))
        );

        var frames = new[] { frame1, frame2 };
        var tempFile = Path.GetTempFileName();

        try
        {
            // Act & Assert - should not throw
            Assert.DoesNotThrow(() => ParquetWriter.WriteParquetBatch(frames, tempFile));

            // Verify file was created
            Assert.That(File.Exists(tempFile), Is.True);
            Assert.That(new FileInfo(tempFile).Length, Is.GreaterThan(0));
        }
        finally
        {
            // Cleanup
            frame1.Dispose();
            frame2.Dispose();
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public void WriteParquetBatch_WithIncompatibleSchemas_ThrowsSchemaValidationException()
    {
        // Arrange
        var frame1 = NivaraFrame.Create(
            ("IntColumn", NivaraColumn<int>.Create(new[] { 1, 2 }))
        );

        var frame2 = NivaraFrame.Create(
            ("StringColumn", NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b" }))
        );

        var frames = new[] { frame1, frame2 };
        var tempFile = Path.GetTempFileName();

        try
        {
            // Act & Assert
            Assert.Throws<SchemaValidationException>(() => ParquetWriter.WriteParquetBatch(frames, tempFile));
        }
        finally
        {
            // Cleanup
            frame1.Dispose();
            frame2.Dispose();
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public void WriteParquet_WithNullableValues_HandlesNullsCorrectly()
    {
        // Arrange
        var nullableIntArray = new int?[] { 1, null, 3, null, 5 };
        var intColumn = NivaraColumn<int>.CreateFromNullable(nullableIntArray);

        var stringArray = new string[] { "a", null!, "c", null!, "e" };
        var stringColumn = NivaraColumn<string>.CreateForReferenceType(stringArray);

        var frame = NivaraFrame.Create(
            ("NullableIntColumn", intColumn),
            ("NullableStringColumn", stringColumn)
        );

        var tempFile = Path.GetTempFileName();

        try
        {
            // Act & Assert - should not throw
            Assert.DoesNotThrow(() => ParquetWriter.WriteParquet(frame, tempFile));

            // Verify file was created
            Assert.That(File.Exists(tempFile), Is.True);
            Assert.That(new FileInfo(tempFile).Length, Is.GreaterThan(0));
        }
        finally
        {
            // Cleanup
            frame.Dispose();
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public void WriteParquet_WithDateTimeColumn_WritesSuccessfully()
    {
        // Arrange
        var dateTimeColumn = NivaraColumn<DateTime>.Create(new[]
        {
            new DateTime(2023, 1, 1),
            new DateTime(2023, 6, 15),
            new DateTime(2023, 12, 31)
        });

        var frame = NivaraFrame.Create(("DateTimeColumn", dateTimeColumn));
        var tempFile = Path.GetTempFileName();

        try
        {
            // Act & Assert - should not throw
            Assert.DoesNotThrow(() => ParquetWriter.WriteParquet(frame, tempFile));

            // Verify file was created
            Assert.That(File.Exists(tempFile), Is.True);
            Assert.That(new FileInfo(tempFile).Length, Is.GreaterThan(0));
        }
        finally
        {
            // Cleanup
            frame.Dispose();
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }
}