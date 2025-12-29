using NUnit.Framework;

namespace Nivara.Tests;

[TestFixture]
public class ExecutionContextTests
{
    [Test]
    public void Constructor_DefaultValues_SetsExpectedDefaults()
    {
        // Act
        var context = new ExecutionContext();

        // Assert
        Assert.That(context.Strategy, Is.EqualTo(ExecutionStrategy.Lazy));
        Assert.That(context.MaxDegreeOfParallelism, Is.EqualTo(Environment.ProcessorCount));
        Assert.That(context.MemoryBudget, Is.EqualTo(1024 * 1024 * 1024)); // 1GB
        Assert.That(context.CancellationToken, Is.EqualTo(CancellationToken.None));
        Assert.That(context.Progress, Is.Null);
    }

    [Test]
    public void Constructor_WithStrategy_SetsStrategy()
    {
        // Act
        var context = new ExecutionContext(ExecutionStrategy.Parallel);

        // Assert
        Assert.That(context.Strategy, Is.EqualTo(ExecutionStrategy.Parallel));
        Assert.That(context.MaxDegreeOfParallelism, Is.EqualTo(Environment.ProcessorCount));
    }

    [Test]
    public void Clone_CreatesIdenticalCopy()
    {
        // Arrange
        var original = new ExecutionContext(ExecutionStrategy.Streaming)
        {
            MaxDegreeOfParallelism = 4,
            MemoryBudget = 512 * 1024 * 1024
        };

        // Act
        var clone = original.Clone();

        // Assert
        Assert.That(clone.Strategy, Is.EqualTo(original.Strategy));
        Assert.That(clone.MaxDegreeOfParallelism, Is.EqualTo(original.MaxDegreeOfParallelism));
        Assert.That(clone.MemoryBudget, Is.EqualTo(original.MemoryBudget));
        Assert.That(clone.CancellationToken, Is.EqualTo(original.CancellationToken));
        Assert.That(clone, Is.Not.SameAs(original)); // Different instances
    }

    [Test]
    public void WithStrategy_CreatesContextWithSpecifiedStrategy()
    {
        // Act
        var context = ExecutionContext.WithStrategy(ExecutionStrategy.Eager);

        // Assert
        Assert.That(context.Strategy, Is.EqualTo(ExecutionStrategy.Eager));
    }

    [Test]
    public void WithParallelism_CreatesParallelContextWithSpecifiedDegree()
    {
        // Act
        var context = ExecutionContext.WithParallelism(8);

        // Assert
        Assert.That(context.Strategy, Is.EqualTo(ExecutionStrategy.Parallel));
        Assert.That(context.MaxDegreeOfParallelism, Is.EqualTo(8));
    }

    [Test]
    public void WithMemoryBudget_CreatesContextWithSpecifiedBudget()
    {
        // Arrange
        var budgetBytes = 256 * 1024 * 1024; // 256MB

        // Act
        var context = ExecutionContext.WithMemoryBudget(budgetBytes);

        // Assert
        Assert.That(context.MemoryBudget, Is.EqualTo(budgetBytes));
    }

    [Test]
    public void WithCancellation_CreatesContextWithSpecifiedToken()
    {
        // Arrange
        using var cts = new CancellationTokenSource();
        var token = cts.Token;

        // Act
        var context = ExecutionContext.WithCancellation(token);

        // Assert
        Assert.That(context.CancellationToken, Is.EqualTo(token));
    }

    [Test]
    public void ToString_ReturnsFormattedString()
    {
        // Arrange
        var context = new ExecutionContext(ExecutionStrategy.Parallel)
        {
            MaxDegreeOfParallelism = 4,
            MemoryBudget = 512 * 1024 * 1024
        };

        // Act
        var result = context.ToString();

        // Assert
        Assert.That(result, Does.Contain("ExecutionContext"));
        Assert.That(result, Does.Contain("Strategy: Parallel"));
        Assert.That(result, Does.Contain("MaxDegreeOfParallelism: 4"));
        Assert.That(result, Does.Contain("MemoryBudget: 536,870,912 bytes"));
    }
}

[TestFixture]
public class ExecutionProgressTests
{
    [Test]
    public void Constructor_ValidParameters_SetsProperties()
    {
        // Act
        var progress = new ExecutionProgress("TestOperation", 50, 100);

        // Assert
        Assert.That(progress.OperationName, Is.EqualTo("TestOperation"));
        Assert.That(progress.CompletedWork, Is.EqualTo(50));
        Assert.That(progress.TotalWork, Is.EqualTo(100));
    }

    [Test]
    public void Constructor_NullOperationName_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new ExecutionProgress(null!, 50, 100));
    }

    [Test]
    public void PercentComplete_CalculatesCorrectPercentage()
    {
        // Arrange
        var progress = new ExecutionProgress("Test", 25, 100);

        // Act
        var percent = progress.PercentComplete;

        // Assert
        Assert.That(percent, Is.EqualTo(0.25).Within(0.001));
    }

    [Test]
    public void PercentComplete_ZeroTotalWork_ReturnsZero()
    {
        // Arrange
        var progress = new ExecutionProgress("Test", 50, 0);

        // Act
        var percent = progress.PercentComplete;

        // Assert
        Assert.That(percent, Is.EqualTo(0.0));
    }

    [Test]
    public void IsComplete_CompletedEqualsTotal_ReturnsTrue()
    {
        // Arrange
        var progress = new ExecutionProgress("Test", 100, 100);

        // Act
        var isComplete = progress.IsComplete;

        // Assert
        Assert.That(isComplete, Is.True);
    }

    [Test]
    public void IsComplete_CompletedLessThanTotal_ReturnsFalse()
    {
        // Arrange
        var progress = new ExecutionProgress("Test", 50, 100);

        // Act
        var isComplete = progress.IsComplete;

        // Assert
        Assert.That(isComplete, Is.False);
    }

    [Test]
    public void IsComplete_CompletedGreaterThanTotal_ReturnsTrue()
    {
        // Arrange
        var progress = new ExecutionProgress("Test", 150, 100);

        // Act
        var isComplete = progress.IsComplete;

        // Assert
        Assert.That(isComplete, Is.True);
    }

    [Test]
    public void ToString_ReturnsFormattedString()
    {
        // Arrange
        var progress = new ExecutionProgress("TestOperation", 75, 100);

        // Act
        var result = progress.ToString();

        // Assert
        Assert.That(result, Is.EqualTo("TestOperation: 75/100 (75.0%)"));
    }
}