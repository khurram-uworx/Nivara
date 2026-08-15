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
        var options = ParquetWriteOptions.Default.With(validateSchema: true);

        try
        {
            // Act & Assert - should not throw
            Assert.DoesNotThrow(() => NivaraParquetWriter.WriteParquet(frame, tempFile, options));

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
            Assert.DoesNotThrowAsync(async () => await NivaraParquetWriter.WriteParquetAsync(frame, tempFile));

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
            Assert.DoesNotThrow(() => NivaraParquetWriter.WriteParquet(frame, tempFile));

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
            Assert.Throws<ArgumentNullException>(() => NivaraParquetWriter.WriteParquet(null!, tempFile));
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
            Assert.Throws<ArgumentNullException>(() => NivaraParquetWriter.WriteParquet(frame, (string)null!));
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
        var options = ParquetWriteOptions.Default;

        try
        {
            // Act & Assert - should not throw
            Assert.DoesNotThrow(() => NivaraParquetWriter.WriteParquet(frame, stream, options));

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
            Assert.DoesNotThrow(() => NivaraParquetWriter.WriteParquetBatch(frames, tempFile));

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
            Assert.Throws<SchemaValidationException>(() => NivaraParquetWriter.WriteParquetBatch(frames, tempFile));
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
        var intColumn = NivaraColumn.CreateFromNullable(nullableIntArray);

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
            Assert.DoesNotThrow(() => NivaraParquetWriter.WriteParquet(frame, tempFile));

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
            Assert.DoesNotThrow(() => NivaraParquetWriter.WriteParquet(frame, tempFile));

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
    public async Task WriteParquet_WithSmallRowGroupSize_WritesMultipleRowGroupsAndRoundTrips()
    {
        // Arrange
        var values = Enumerable.Range(0, 2500).Select(i => i * 2).ToArray();
        var frame = NivaraFrame.Create(("Index", NivaraColumn<int>.Create(values)));
        var tempFile = Path.GetTempFileName();

        try
        {
            // Act
            NivaraParquetWriter.WriteParquet(frame, tempFile, ParquetWriteOptions.Default.With(rowGroupSize: 1000));

            // Assert - file contains one row group per 1000 rows
            using var fileStream = new FileStream(tempFile, FileMode.Open, FileAccess.Read);
            await using var parquetReader = await Parquet.ParquetReader.CreateAsync(fileStream);
            Assert.That(parquetReader.RowGroupCount, Is.EqualTo(3));

            // All row groups round-trip through the Nivara reader
            var roundTrip = NivaraParquetReader.ReadParquet(tempFile);
            Assert.That(roundTrip.RowCount, Is.EqualTo(2500));
            Assert.That(roundTrip.GetColumn<int>("Index")[2499], Is.EqualTo(4998));
            Assert.That(roundTrip.GetColumn<int>("Index")[1000], Is.EqualTo(2000));
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
    public void WriteParquet_WithGzipCompression_RoundTripsAndCompresses()
    {
        // Arrange
        var strings = Enumerable.Range(0, 5000).Select(i => $"row-{i % 50}").ToArray();
        var frame = NivaraFrame.Create(("Name", NivaraColumn<string>.CreateForReferenceType(strings)));
        var gzipFile = Path.GetTempFileName();
        var noneFile = Path.GetTempFileName();

        try
        {
            // Act
            NivaraParquetWriter.WriteParquet(frame, gzipFile, ParquetWriteOptions.Default.With(compression: ParquetCompression.Gzip));
            NivaraParquetWriter.WriteParquet(frame, noneFile, ParquetWriteOptions.Default.With(compression: ParquetCompression.None));

            // Assert - data round-trips under the configured compression
            var roundTrip = NivaraParquetReader.ReadParquet(gzipFile);
            Assert.That(roundTrip.RowCount, Is.EqualTo(5000));
            Assert.That(roundTrip.GetColumn<string>("Name")[123], Is.EqualTo(strings[123]));

            // Gzip should compress the repetitive data better than no compression
            Assert.That(new FileInfo(gzipFile).Length, Is.LessThan(new FileInfo(noneFile).Length));
        }
        finally
        {
            // Cleanup
            frame.Dispose();
            if (File.Exists(gzipFile))
                File.Delete(gzipFile);
            if (File.Exists(noneFile))
                File.Delete(noneFile);
        }
    }

    [Test]
    public void WriteParquet_WithWriteMetadataDisabled_DoesNotRestoreExtendedTypes()
    {
        // Arrange
        var halfValues = new Half[] { (Half)1.5f, (Half)2.5f, (Half)3.5f };
        var frame = NivaraFrame.Create(("Measurement", NivaraColumn<Half>.Create(halfValues)));
        var noMetadataFile = Path.GetTempFileName();
        var withMetadataFile = Path.GetTempFileName();

        try
        {
            // Act
            NivaraParquetWriter.WriteParquet(frame, noMetadataFile, ParquetWriteOptions.Default.With(writeMetadata: false));
            NivaraParquetWriter.WriteParquet(frame, withMetadataFile);

            // Assert - without metadata the widened on-disk representation is read back
            var noMetadata = NivaraParquetReader.ReadParquet(noMetadataFile);
            Assert.That(noMetadata.Schema.GetColumnType("Measurement"), Is.EqualTo(typeof(float)));

            // With default metadata the original CLR type is restored
            var withMetadata = NivaraParquetReader.ReadParquet(withMetadataFile);
            Assert.That(withMetadata.Schema.GetColumnType("Measurement"), Is.EqualTo(typeof(Half)));
            Assert.That(withMetadata.GetColumn<Half>("Measurement")[0], Is.EqualTo((Half)1.5f));
        }
        finally
        {
            // Cleanup
            frame.Dispose();
            if (File.Exists(noMetadataFile))
                File.Delete(noMetadataFile);
            if (File.Exists(withMetadataFile))
                File.Delete(withMetadataFile);
        }
    }
}
