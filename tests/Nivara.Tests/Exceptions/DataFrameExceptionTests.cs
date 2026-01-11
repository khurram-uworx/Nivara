using NUnit.Framework;
using Nivara.Exceptions;
using Nivara.Operations;

namespace Nivara.Tests.Exceptions;

[TestFixture]
public class DataFrameExceptionTests
{
    [Test]
    public void DataFrameSchemaValidationException_WithSchemas_ProvidesMismatchDetails()
    {
        // Arrange
        var expectedSchema = new Schema(new[]
        {
            ("id", typeof(int)),
            ("name", typeof(string))
        });
        
        var actualSchema = new Schema(new[]
        {
            ("id", typeof(string)), // Type mismatch
            ("age", typeof(int))    // Extra column, missing 'name'
        });

        // Act
        var exception = new DataFrameSchemaValidationException(
            "Schema validation failed", expectedSchema, actualSchema);

        // Assert
        Assert.That(exception.ExpectedSchema, Is.EqualTo(expectedSchema));
        Assert.That(exception.ActualSchema, Is.EqualTo(actualSchema));
        
        var mismatches = exception.Mismatches;
        Assert.That(mismatches.Count, Is.EqualTo(3)); // Missing 'name', extra 'age', type mismatch 'id'
        
        // Check for missing column
        Assert.That(mismatches.Any(m => m.MismatchType == SchemaMismatchType.MissingColumn && m.ColumnName == "name"), Is.True);
        
        // Check for extra column
        Assert.That(mismatches.Any(m => m.MismatchType == SchemaMismatchType.ExtraColumn && m.ColumnName == "age"), Is.True);
        
        // Check for type mismatch
        Assert.That(mismatches.Any(m => m.MismatchType == SchemaMismatchType.TypeMismatch && m.ColumnName == "id"), Is.True);
    }

    [Test]
    public void JoinException_WithJoinDetails_ProvidesComprehensiveContext()
    {
        // Arrange
        var leftKeys = new[] { "id", "category" };
        var rightKeys = new[] { "user_id", "cat" };
        var conflictReason = "Join key types do not match";

        // Act
        var exception = new JoinException(
            "Join operation failed", 
            JoinType.Inner, 
            leftKeys, 
            rightKeys, 
            conflictReason);

        // Assert
        Assert.That(exception.AttemptedJoinType, Is.EqualTo(JoinType.Inner));
        Assert.That(exception.LeftKeys, Is.EqualTo(leftKeys));
        Assert.That(exception.RightKeys, Is.EqualTo(rightKeys));
        Assert.That(exception.ConflictReason, Is.EqualTo(conflictReason));
        
        var context = exception.GetDetailedContext();
        Assert.That(context, Does.Contain("Join Type: Inner"));
        Assert.That(context, Does.Contain("Left Keys: id, category"));
        Assert.That(context, Does.Contain("Right Keys: user_id, cat"));
        Assert.That(context, Does.Contain("Conflict Reason: Join key types do not match"));
    }

    [Test]
    public void SchemaMismatch_ToString_ProvidesReadableDescription()
    {
        // Arrange
        var mismatch = new SchemaMismatch(
            SchemaMismatchType.TypeMismatch,
            "age",
            typeof(int),
            typeof(string),
            "Column 'age' has type String but expected Int32");

        // Act
        var description = mismatch.ToString();

        // Assert
        Assert.That(description, Is.EqualTo("Column 'age' has type String but expected Int32"));
    }

    [Test]
    public void DataFrameSchemaValidationException_GetDetailedContext_IncludesAllRelevantInformation()
    {
        // Arrange
        var expectedSchema = new Schema(new[]
        {
            ("id", typeof(int)),
            ("name", typeof(string))
        });
        
        var actualSchema = new Schema(new[]
        {
            ("id", typeof(int)),
            ("age", typeof(int))
        });

        var exception = new DataFrameSchemaValidationException(
            "Schema mismatch detected", expectedSchema, actualSchema);

        // Act
        var context = exception.GetDetailedContext();

        // Assert
        Assert.That(context, Does.Contain("DataFrameSchemaValidationException"));
        Assert.That(context, Does.Contain("Schema mismatch detected"));
        Assert.That(context, Does.Contain("Expected Schema:"));
        Assert.That(context, Does.Contain("Actual Schema:"));
        Assert.That(context, Does.Contain("Schema Mismatches:"));
    }

    [Test]
    public void JoinException_WithoutOptionalParameters_HandlesGracefully()
    {
        // Act
        var exception = new JoinException("Simple join failure");

        // Assert
        Assert.That(exception.Message, Is.EqualTo("Simple join failure"));
        Assert.That(exception.LeftKeys, Is.Empty);
        Assert.That(exception.RightKeys, Is.Empty);
        Assert.That(exception.ConflictReason, Is.Null);
    }
}