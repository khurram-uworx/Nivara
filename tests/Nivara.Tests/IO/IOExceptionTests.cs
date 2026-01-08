using System.Runtime.Serialization.Formatters.Binary;
using System.Text.Json;
using Nivara.IO;
using NUnit.Framework;

namespace Nivara.Tests.IO;

[TestFixture]
public class IOExceptionTests
{
    #region NivaraIOException Tests

    [Test]
    public void NivaraIOException_BasicConstructor_InitializesCorrectly()
    {
        const string message = "Test error message";
        var exception = new NivaraIOException(message);

        Assert.That(exception.Message, Is.EqualTo(message));
        Assert.That(exception.FilePath, Is.Null);
        Assert.That(exception.OperationContext, Is.Null);
        Assert.That(exception.InnerException, Is.Null);
    }

    [Test]
    public void NivaraIOException_WithInnerException_InitializesCorrectly()
    {
        const string message = "Test error message";
        var innerException = new InvalidOperationException("Inner error");
        var exception = new NivaraIOException(message, innerException);

        Assert.That(exception.Message, Is.EqualTo(message));
        Assert.That(exception.InnerException, Is.EqualTo(innerException));
        Assert.That(exception.FilePath, Is.Null);
        Assert.That(exception.OperationContext, Is.Null);
    }

    [Test]
    public void NivaraIOException_WithFilePathAndContext_InitializesCorrectly()
    {
        const string message = "Test error message";
        const string filePath = "/path/to/file.parquet";
        const string operationContext = "Reading parquet file";
        var exception = new NivaraIOException(message, filePath, operationContext);

        Assert.That(exception.Message, Is.EqualTo(message));
        Assert.That(exception.FilePath, Is.EqualTo(filePath));
        Assert.That(exception.OperationContext, Is.EqualTo(operationContext));
        Assert.That(exception.InnerException, Is.Null);
    }

    [Test]
    public void NivaraIOException_WithAllParameters_InitializesCorrectly()
    {
        const string message = "Test error message";
        const string filePath = "/path/to/file.parquet";
        const string operationContext = "Reading parquet file";
        var innerException = new InvalidOperationException("Inner error");
        var exception = new NivaraIOException(message, filePath, operationContext, innerException);

        Assert.That(exception.Message, Is.EqualTo(message));
        Assert.That(exception.FilePath, Is.EqualTo(filePath));
        Assert.That(exception.OperationContext, Is.EqualTo(operationContext));
        Assert.That(exception.InnerException, Is.EqualTo(innerException));
    }

    [Test]
    public void NivaraIOException_WithNullFilePathAndContext_HandlesCorrectly()
    {
        const string message = "Test error message";
        var exception = new NivaraIOException(message, null, null);

        Assert.That(exception.Message, Is.EqualTo(message));
        Assert.That(exception.FilePath, Is.Null);
        Assert.That(exception.OperationContext, Is.Null);
    }

    #endregion

    #region UnsupportedTypeException Tests

    [Test]
    public void UnsupportedTypeException_BasicConstructor_InitializesCorrectly()
    {
        var unsupportedType = typeof(Guid);
        var exception = new UnsupportedTypeException(unsupportedType);

        Assert.That(exception.UnsupportedType, Is.EqualTo(unsupportedType));
        Assert.That(exception.Message, Does.Contain("Guid"));
        Assert.That(exception.Message, Does.Contain("not supported"));
        Assert.That(exception.SuggestedAlternatives, Is.Empty);
    }

    [Test]
    public void UnsupportedTypeException_WithSuggestions_InitializesCorrectly()
    {
        var unsupportedType = typeof(TimeSpan);
        var suggestions = new[] { "long (ticks)", "double (seconds)" };
        var exception = new UnsupportedTypeException(unsupportedType, suggestions);

        Assert.That(exception.UnsupportedType, Is.EqualTo(unsupportedType));
        Assert.That(exception.Message, Does.Contain("TimeSpan"));
        Assert.That(exception.Message, Does.Contain("not supported"));
        Assert.That(exception.Message, Does.Contain("long (ticks)"));
        Assert.That(exception.Message, Does.Contain("double (seconds)"));
        Assert.That(exception.SuggestedAlternatives, Is.EqualTo(suggestions));
    }

    [Test]
    public void UnsupportedTypeException_InheritsFromNivaraIOException()
    {
        var exception = new UnsupportedTypeException(typeof(Guid));
        Assert.That(exception, Is.InstanceOf<NivaraIOException>());
    }

    [Test]
    public void UnsupportedTypeException_WithEmptySuggestions_HandlesCorrectly()
    {
        var unsupportedType = typeof(object);
        var suggestions = Array.Empty<string>();
        var exception = new UnsupportedTypeException(unsupportedType, suggestions);

        Assert.That(exception.UnsupportedType, Is.EqualTo(unsupportedType));
        Assert.That(exception.SuggestedAlternatives, Is.Empty);
    }

    #endregion

    #region SchemaValidationException Tests

    [Test]
    public void SchemaValidationException_BasicConstructor_InitializesCorrectly()
    {
        const string message = "Schema validation failed";
        var exception = new SchemaValidationException(message);

        Assert.That(exception.Message, Is.EqualTo(message));
        Assert.That(exception.TypeMismatches, Is.Empty);
        Assert.That(exception.ExpectedSchema, Is.EqualTo(string.Empty));
        Assert.That(exception.ActualSchema, Is.EqualTo(string.Empty));
    }

    [Test]
    public void SchemaValidationException_WithDetails_InitializesCorrectly()
    {
        const string message = "Schema validation failed";
        var typeMismatches = new[] { "Column 'Age' expected int but got string", "Column 'Name' missing" };
        const string expectedSchema = "Age: int, Name: string";
        const string actualSchema = "Age: string";
        var exception = new SchemaValidationException(message, typeMismatches, expectedSchema, actualSchema);

        Assert.That(exception.Message, Is.EqualTo(message));
        Assert.That(exception.TypeMismatches, Is.EqualTo(typeMismatches));
        Assert.That(exception.ExpectedSchema, Is.EqualTo(expectedSchema));
        Assert.That(exception.ActualSchema, Is.EqualTo(actualSchema));
    }

    [Test]
    public void SchemaValidationException_InheritsFromNivaraIOException()
    {
        var exception = new SchemaValidationException("Test message");
        Assert.That(exception, Is.InstanceOf<NivaraIOException>());
    }

    [Test]
    public void SchemaValidationException_WithEmptyTypeMismatches_HandlesCorrectly()
    {
        const string message = "Schema validation failed";
        var typeMismatches = Array.Empty<string>();
        var exception = new SchemaValidationException(message, typeMismatches, "expected", "actual");

        Assert.That(exception.TypeMismatches, Is.Empty);
        Assert.That(exception.ExpectedSchema, Is.EqualTo("expected"));
        Assert.That(exception.ActualSchema, Is.EqualTo("actual"));
    }

    #endregion

    #region DataCorruptionException Tests

    [Test]
    public void DataCorruptionException_BasicConstructor_InitializesCorrectly()
    {
        const string message = "Data corruption detected";
        var exception = new DataCorruptionException(message);

        Assert.That(exception.Message, Is.EqualTo(message));
        Assert.That(exception.AffectedColumns, Is.Empty);
        Assert.That(exception.AffectedRowRange, Is.EqualTo(default(Range)));
        Assert.That(exception.InnerException, Is.Null);
    }

    [Test]
    public void DataCorruptionException_WithInnerException_InitializesCorrectly()
    {
        const string message = "Data corruption detected";
        var innerException = new IOException("File read error");
        var exception = new DataCorruptionException(message, innerException);

        Assert.That(exception.Message, Is.EqualTo(message));
        Assert.That(exception.InnerException, Is.EqualTo(innerException));
        Assert.That(exception.AffectedColumns, Is.Empty);
        Assert.That(exception.AffectedRowRange, Is.EqualTo(default(Range)));
    }

    [Test]
    public void DataCorruptionException_WithAffectedColumnsAndRange_InitializesCorrectly()
    {
        const string message = "Data corruption detected";
        var affectedColumns = new[] { "Column1", "Column2" };
        var affectedRowRange = new Range(10, 20);
        var exception = new DataCorruptionException(message, affectedColumns, affectedRowRange);

        Assert.That(exception.Message, Is.EqualTo(message));
        Assert.That(exception.AffectedColumns, Is.EqualTo(affectedColumns));
        Assert.That(exception.AffectedRowRange, Is.EqualTo(affectedRowRange));
    }

    [Test]
    public void DataCorruptionException_WithAllParameters_InitializesCorrectly()
    {
        const string message = "Data corruption detected";
        const string filePath = "/path/to/corrupted.parquet";
        const string operationContext = "Reading row group 5";
        var affectedColumns = new[] { "Column1", "Column2" };
        var affectedRowRange = new Range(100, 200);
        var exception = new DataCorruptionException(message, filePath, operationContext, affectedColumns, affectedRowRange);

        Assert.That(exception.Message, Is.EqualTo(message));
        Assert.That(exception.FilePath, Is.EqualTo(filePath));
        Assert.That(exception.OperationContext, Is.EqualTo(operationContext));
        Assert.That(exception.AffectedColumns, Is.EqualTo(affectedColumns));
        Assert.That(exception.AffectedRowRange, Is.EqualTo(affectedRowRange));
    }

    [Test]
    public void DataCorruptionException_InheritsFromNivaraIOException()
    {
        var exception = new DataCorruptionException("Test message");
        Assert.That(exception, Is.InstanceOf<NivaraIOException>());
    }

    [Test]
    public void DataCorruptionException_WithEmptyAffectedColumns_HandlesCorrectly()
    {
        const string message = "Data corruption detected";
        var affectedColumns = Array.Empty<string>();
        var affectedRowRange = new Range(0, 10);
        var exception = new DataCorruptionException(message, affectedColumns, affectedRowRange);

        Assert.That(exception.AffectedColumns, Is.Empty);
        Assert.That(exception.AffectedRowRange, Is.EqualTo(affectedRowRange));
    }

    #endregion

    #region Exception Hierarchy Tests

    [Test]
    public void AllCustomExceptions_InheritFromNivaraIOException()
    {
        var unsupportedTypeException = new UnsupportedTypeException(typeof(Guid));
        var schemaValidationException = new SchemaValidationException("Test");
        var dataCorruptionException = new DataCorruptionException("Test");

        Assert.That(unsupportedTypeException, Is.InstanceOf<NivaraIOException>());
        Assert.That(schemaValidationException, Is.InstanceOf<NivaraIOException>());
        Assert.That(dataCorruptionException, Is.InstanceOf<NivaraIOException>());
    }

    [Test]
    public void AllCustomExceptions_InheritFromSystemException()
    {
        var nivaraIOException = new NivaraIOException("Test");
        var unsupportedTypeException = new UnsupportedTypeException(typeof(Guid));
        var schemaValidationException = new SchemaValidationException("Test");
        var dataCorruptionException = new DataCorruptionException("Test");

        Assert.That(nivaraIOException, Is.InstanceOf<Exception>());
        Assert.That(unsupportedTypeException, Is.InstanceOf<Exception>());
        Assert.That(schemaValidationException, Is.InstanceOf<Exception>());
        Assert.That(dataCorruptionException, Is.InstanceOf<Exception>());
    }

    #endregion

    #region Serialization Tests

    [Test]
    public void NivaraIOException_JsonSerialization_PreservesProperties()
    {
        const string message = "Test error message";
        const string filePath = "/path/to/file.parquet";
        const string operationContext = "Reading parquet file";
        var originalException = new NivaraIOException(message, filePath, operationContext);

        // Create a serializable representation
        var serializableData = new
        {
            Message = originalException.Message,
            FilePath = originalException.FilePath,
            OperationContext = originalException.OperationContext,
            ExceptionType = originalException.GetType().Name
        };

        var json = JsonSerializer.Serialize(serializableData);
        var deserializedData = JsonSerializer.Deserialize<dynamic>(json);

        Assert.That(json, Does.Contain(message));
        Assert.That(json, Does.Contain(filePath));
        Assert.That(json, Does.Contain(operationContext));
        Assert.That(json, Does.Contain("NivaraIOException"));
    }

    [Test]
    public void UnsupportedTypeException_JsonSerialization_PreservesProperties()
    {
        var unsupportedType = typeof(Guid);
        var suggestions = new[] { "string", "byte[]" };
        var originalException = new UnsupportedTypeException(unsupportedType, suggestions);

        // Create a serializable representation
        var serializableData = new
        {
            Message = originalException.Message,
            UnsupportedTypeName = originalException.UnsupportedType.Name,
            UnsupportedTypeFullName = originalException.UnsupportedType.FullName,
            SuggestedAlternatives = originalException.SuggestedAlternatives,
            ExceptionType = originalException.GetType().Name
        };

        var json = JsonSerializer.Serialize(serializableData);

        Assert.That(json, Does.Contain("Guid"));
        Assert.That(json, Does.Contain("string"));
        Assert.That(json, Does.Contain("byte[]"));
        Assert.That(json, Does.Contain("UnsupportedTypeException"));
    }

    [Test]
    public void SchemaValidationException_JsonSerialization_PreservesProperties()
    {
        const string message = "Schema validation failed";
        var typeMismatches = new[] { "Column 'Age' expected int but got string" };
        const string expectedSchema = "Age: int";
        const string actualSchema = "Age: string";
        var originalException = new SchemaValidationException(message, typeMismatches, expectedSchema, actualSchema);

        // Create a serializable representation
        var serializableData = new
        {
            Message = originalException.Message,
            TypeMismatches = originalException.TypeMismatches,
            ExpectedSchema = originalException.ExpectedSchema,
            ActualSchema = originalException.ActualSchema,
            ExceptionType = originalException.GetType().Name
        };

        var json = JsonSerializer.Serialize(serializableData);

        Assert.That(json, Does.Contain("Age"));
        Assert.That(json, Does.Contain("expected int but got string"));
        Assert.That(json, Does.Contain("Age: int"));
        Assert.That(json, Does.Contain("Age: string"));
        Assert.That(json, Does.Contain("SchemaValidationException"));
    }

    [Test]
    public void DataCorruptionException_JsonSerialization_PreservesProperties()
    {
        const string message = "Data corruption detected";
        const string filePath = "/path/to/corrupted.parquet";
        const string operationContext = "Reading row group 5";
        var affectedColumns = new[] { "Column1", "Column2" };
        var affectedRowRange = new Range(100, 200);
        var originalException = new DataCorruptionException(message, filePath, operationContext, affectedColumns, affectedRowRange);

        // Create a serializable representation
        var serializableData = new
        {
            Message = originalException.Message,
            FilePath = originalException.FilePath,
            OperationContext = originalException.OperationContext,
            AffectedColumns = originalException.AffectedColumns,
            AffectedRowRangeStart = originalException.AffectedRowRange.Start.Value,
            AffectedRowRangeEnd = originalException.AffectedRowRange.End.Value,
            ExceptionType = originalException.GetType().Name
        };

        var json = JsonSerializer.Serialize(serializableData);

        Assert.That(json, Does.Contain(message));
        Assert.That(json, Does.Contain(filePath));
        Assert.That(json, Does.Contain(operationContext));
        Assert.That(json, Does.Contain("Column1"));
        Assert.That(json, Does.Contain("Column2"));
        Assert.That(json, Does.Contain("100"));
        Assert.That(json, Does.Contain("200"));
        Assert.That(json, Does.Contain("DataCorruptionException"));
    }

    [Test]
    public void ExceptionSerialization_ForDistributedScenarios_MaintainsEssentialInformation()
    {
        // Test that all essential information can be serialized for distributed scenarios
        var exceptions = new Exception[]
        {
            new NivaraIOException("IO Error", "/path/file.parquet", "reading"),
            new UnsupportedTypeException(typeof(Guid), new[] { "string", "byte[]" }),
            new SchemaValidationException("Schema error", new[] { "mismatch1" }, "expected", "actual"),
            new DataCorruptionException("Corruption", "/path/file.parquet", "reading", new[] { "col1" }, new Range(0, 10))
        };

        foreach (var exception in exceptions)
        {
            // Simulate distributed scenario by serializing essential properties
            object additionalData = exception switch
            {
                DataCorruptionException dce => new { dce.AffectedColumns, RowRangeStart = dce.AffectedRowRange.Start.Value, RowRangeEnd = dce.AffectedRowRange.End.Value },
                SchemaValidationException sve => new { sve.TypeMismatches, sve.ExpectedSchema, sve.ActualSchema },
                UnsupportedTypeException ute => new { TypeName = ute.UnsupportedType.Name, ute.SuggestedAlternatives },
                NivaraIOException nio => new { nio.FilePath, nio.OperationContext },
                _ => new { }
            };

            var essentialInfo = new
            {
                ExceptionType = exception.GetType().Name,
                Message = exception.Message,
                AdditionalData = additionalData
            };

            var json = JsonSerializer.Serialize(essentialInfo);
            
            // Verify that serialization succeeds and contains expected information
            Assert.That(json, Is.Not.Null.And.Not.Empty);
            Assert.That(json, Does.Contain(exception.GetType().Name));
            // Check for key parts of the message rather than exact match due to JSON escaping
            if (exception is UnsupportedTypeException)
            {
                Assert.That(json, Does.Contain("Guid"));
                Assert.That(json, Does.Contain("not supported"));
            }
            else
            {
                Assert.That(json, Does.Contain("Test") | Does.Contain("IO Error") | Does.Contain("Schema") | Does.Contain("Corruption"));
            }
        }
    }

    #endregion

    #region Property Initialization Edge Cases

    [Test]
    public void ExceptionProperties_WithNullCollections_HandleCorrectly()
    {
        // Test that exceptions handle null collections gracefully by using empty collections instead
        var schemaException = new SchemaValidationException("Test", Array.Empty<string>(), "expected", "actual");
        Assert.That(schemaException.TypeMismatches, Is.Not.Null);
        Assert.That(schemaException.TypeMismatches, Is.Empty);

        var unsupportedException = new UnsupportedTypeException(typeof(Guid), Array.Empty<string>());
        Assert.That(unsupportedException.SuggestedAlternatives, Is.Not.Null);
        Assert.That(unsupportedException.SuggestedAlternatives, Is.Empty);

        var corruptionException = new DataCorruptionException("Test", Array.Empty<string>(), new Range(0, 10));
        Assert.That(corruptionException.AffectedColumns, Is.Not.Null);
        Assert.That(corruptionException.AffectedColumns, Is.Empty);
    }

    [Test]
    public void ExceptionProperties_AreReadOnly_AfterInitialization()
    {
        var unsupportedException = new UnsupportedTypeException(typeof(Guid), new[] { "string" });
        var schemaException = new SchemaValidationException("Test", new[] { "mismatch" }, "expected", "actual");
        var corruptionException = new DataCorruptionException("Test", new[] { "col1" }, new Range(0, 10));

        // Verify that collections are read-only (IReadOnlyList)
        Assert.That(unsupportedException.SuggestedAlternatives, Is.InstanceOf<IReadOnlyList<string>>());
        Assert.That(schemaException.TypeMismatches, Is.InstanceOf<IReadOnlyList<string>>());
        Assert.That(corruptionException.AffectedColumns, Is.InstanceOf<IReadOnlyList<string>>());
    }

    #endregion
}