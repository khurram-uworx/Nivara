using NUnit.Framework;
using Nivara.IO;
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace Nivara.Tests.IO;

[TestFixture]
public class ParquetReaderTests
{
    [Test]
    public void ReadParquet_NullFilePath_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => ParquetReader.ReadParquet((string)null!));
    }

    [Test]
    public void ReadParquetAsync_NullFilePath_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.ThrowsAsync<ArgumentNullException>(async () => await ParquetReader.ReadParquetAsync((string)null!));
    }

    [Test]
    public void ReadParquet_NullStream_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => ParquetReader.ReadParquet((Stream)null!));
    }

    [Test]
    public void ReadParquetAsync_NullStream_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.ThrowsAsync<ArgumentNullException>(async () => await ParquetReader.ReadParquetAsync((Stream)null!));
    }

    [Test]
    public void ReadParquet_NonExistentFile_ThrowsFileNotFoundException()
    {
        // Arrange
        var nonExistentPath = "non_existent_file.parquet";

        // Act & Assert
        Assert.Throws<FileNotFoundException>(() => ParquetReader.ReadParquet(nonExistentPath));
    }

    [Test]
    public void ReadParquetAsync_NonExistentFile_ThrowsFileNotFoundException()
    {
        // Arrange
        var nonExistentPath = "non_existent_file.parquet";

        // Act & Assert
        Assert.ThrowsAsync<FileNotFoundException>(async () => await ParquetReader.ReadParquetAsync(nonExistentPath));
    }

    [Test]
    public void ReadParquetStreaming_NullFilePath_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => ParquetReader.ReadParquetStreaming((string)null!).ToList());
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