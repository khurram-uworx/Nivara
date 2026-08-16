using Nivara.IO;
using NUnit.Framework;

namespace Nivara.Tests.IO;

[TestFixture]
public class ParquetReaderTests
{
    [Test]
    public void ReadParquet_NullFilePath_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => NivaraParquetReader.ReadParquet((string)null!));
    }

    [Test]
    public void ReadParquetAsync_NullFilePath_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.ThrowsAsync<ArgumentNullException>(async () => await NivaraParquetReader.ReadParquetAsync((string)null!));
    }

    [Test]
    public void ReadParquet_NullStream_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => NivaraParquetReader.ReadParquet((Stream)null!));
    }

    [Test]
    public void ReadParquetAsync_NullStream_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.ThrowsAsync<ArgumentNullException>(async () => await NivaraParquetReader.ReadParquetAsync((Stream)null!));
    }

    [Test]
    public void ReadParquet_NonExistentFile_ThrowsFileNotFoundException()
    {
        // Arrange
        var nonExistentPath = "non_existent_file.parquet";

        // Act & Assert
        Assert.Throws<FileNotFoundException>(() => NivaraParquetReader.ReadParquet(nonExistentPath));
    }

    [Test]
    public void ReadParquetAsync_NonExistentFile_ThrowsFileNotFoundException()
    {
        // Arrange
        var nonExistentPath = "non_existent_file.parquet";

        // Act & Assert
        Assert.ThrowsAsync<FileNotFoundException>(async () => await NivaraParquetReader.ReadParquetAsync(nonExistentPath));
    }

    [Test]
    public void ReadParquetStreaming_NullFilePath_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => NivaraParquetReader.ReadParquetStreaming((string)null!).ToList());
    }

    [Test]
    public void ParquetReadOptions_DefaultValues_AreCorrect()
    {
        // Arrange & Act
        var options = new ParquetReadOptions();

        // Assert
        Assert.That(options.StreamRowGroups, Is.False);
        Assert.That(options.BatchSize, Is.EqualTo(1000));
        Assert.That(options.ValidateSchema, Is.True);
    }
}
