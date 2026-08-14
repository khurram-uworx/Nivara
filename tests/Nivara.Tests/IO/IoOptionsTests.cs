using Nivara.IO;
using NUnit.Framework;
using System.Globalization;
using System.Text.Json;

namespace Nivara.Tests.IO;

[TestFixture]
public class IoOptionsTests
{
    [Test]
    public void JsonOptions_Default_HasExpectedValues()
    {
        var options = JsonOptions.Default;

        Assert.That(options, Is.Not.Null);
        Assert.That(options.SchemaInferenceRecords, Is.EqualTo(100));
        Assert.That(options.IsArray, Is.True);
        Assert.That(options.SerializerOptions, Is.Not.Null);
        Assert.That(options.SerializerOptions.PropertyNameCaseInsensitive, Is.True);
        Assert.That(options.SerializerOptions.AllowTrailingCommas, Is.True);
    }

    [Test]
    public void JsonOptions_With_ReturnsNewInstancePreservingUnchangedValues()
    {
        var options = JsonOptions.Default.With(schemaInferenceRecords: 500);

        Assert.That(options, Is.Not.SameAs(JsonOptions.Default));
        Assert.That(options.SchemaInferenceRecords, Is.EqualTo(500));
        Assert.That(options.IsArray, Is.True);
        Assert.That(JsonOptions.Default.SchemaInferenceRecords, Is.EqualTo(100));
    }

    [Test]
    public void JsonOptions_With_ClonesSerializerOptionsToPreventAliasing()
    {
        var serializerOptions = new JsonSerializerOptions { PropertyNameCaseInsensitive = false };

        var options = JsonOptions.Default.With(serializerOptions: serializerOptions);

        serializerOptions.PropertyNameCaseInsensitive = true;
        Assert.That(options.SerializerOptions, Is.Not.SameAs(serializerOptions));
        Assert.That(options.SerializerOptions.PropertyNameCaseInsensitive, Is.False);
    }

    [Test]
    public void CsvOptions_Default_HasExpectedValues()
    {
        var options = CsvOptions.Default;

        Assert.That(options, Is.Not.Null);
        Assert.That(options.HasHeaderRecord, Is.True);
        Assert.That(options.Delimiter, Is.EqualTo(","));
        Assert.That(options.Culture, Is.EqualTo(CultureInfo.InvariantCulture));
        Assert.That(options.SchemaInferenceRecords, Is.EqualTo(100));
        Assert.That(options.IgnoreBlankLines, Is.True);
        Assert.That(options.TrimOptions, Is.EqualTo(CsvTrimOptions.Trim));
    }

    [Test]
    public void CsvOptions_With_ReturnsNewInstancePreservingUnchangedValues()
    {
        var options = CsvOptions.Default.With(delimiter: ";", hasHeaderRecord: false);

        Assert.That(options, Is.Not.SameAs(CsvOptions.Default));
        Assert.That(options.Delimiter, Is.EqualTo(";"));
        Assert.That(options.HasHeaderRecord, Is.False);
        Assert.That(options.SchemaInferenceRecords, Is.EqualTo(100));
        Assert.That(CsvOptions.Default.Delimiter, Is.EqualTo(","));
    }

    [Test]
    public void CsvOptions_ToCsvConfiguration_MapsTrimOptions()
    {
        var trimmed = CsvOptions.Default.With(trimOptions: CsvTrimOptions.Trim).ToCsvConfiguration();
        Assert.That(trimmed.TrimOptions, Is.EqualTo(CsvHelper.Configuration.TrimOptions.Trim));

        var verbatim = CsvOptions.Default.With(trimOptions: CsvTrimOptions.None).ToCsvConfiguration();
        Assert.That(verbatim.TrimOptions, Is.EqualTo(CsvHelper.Configuration.TrimOptions.None));
    }

    [Test]
    public void ParquetWriteOptions_Default_HasExpectedValues()
    {
        var options = ParquetWriteOptions.Default;

        Assert.That(options, Is.Not.Null);
        Assert.That(options.RowGroupSize, Is.EqualTo(10000));
        Assert.That(options.Compression, Is.EqualTo(ParquetCompression.Snappy));
        Assert.That(options.ValidateSchema, Is.True);
        Assert.That(options.WriteMetadata, Is.True);
    }

    [Test]
    public void ParquetWriteOptions_With_ReturnsNewInstancePreservingUnchangedValues()
    {
        var options = ParquetWriteOptions.Default.With(rowGroupSize: 1000, compression: ParquetCompression.Gzip);

        Assert.That(options, Is.Not.SameAs(ParquetWriteOptions.Default));
        Assert.That(options.RowGroupSize, Is.EqualTo(1000));
        Assert.That(options.Compression, Is.EqualTo(ParquetCompression.Gzip));
        Assert.That(options.ValidateSchema, Is.True);
        Assert.That(options.WriteMetadata, Is.True);
        Assert.That(ParquetWriteOptions.Default.RowGroupSize, Is.EqualTo(10000));
        Assert.That(ParquetWriteOptions.Default.Compression, Is.EqualTo(ParquetCompression.Snappy));
    }
}
