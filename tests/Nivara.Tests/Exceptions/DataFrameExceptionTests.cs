using Nivara.Exceptions;
using Nivara.Operations;
using NUnit.Framework;

namespace Nivara.Tests.Exceptions;

[TestFixture]
public class DataFrameExceptionTests
{
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
