# Contributing to Nivara DataFrame Library

Thank you for your interest in contributing to Nivara! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- .NET 10.0 SDK or later
- Git
- A code editor (Visual Studio, VS Code, or JetBrains Rider recommended)

### Setting Up Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/khurram-uworx/nivara.git
   cd nivara
   ```

2. **Restore dependencies**
   ```bash
   dotnet restore
   ```

3. **Build the project**
   ```bash
   dotnet build
   ```

4. **Run tests**
   ```bash
   dotnet test
   ```

## 📋 Development Process

### Finding Work

2. Look for issues labeled `good first issue` or `help wanted`
3. Review the project status in the [README](README.md#-project-status)

### Implementation Workflow

1. **Choose a Task**: Select an available task from the task list
2. **Review Specifications**: Read the relevant requirements and design sections
3. **Create Branch**: Create a feature branch for your work
4. **Implement**: Follow the coding standards and implement the feature
5. **Test**: Add comprehensive unit and property-based tests
6. **Document**: Update documentation and examples as needed
7. **Submit PR**: Create a pull request with clear description

## 🧪 Testing Guidelines

### Test Requirements

All contributions must include appropriate tests:

- **Unit Tests**: Specific examples and edge cases using NUnit
- **Property-Based Tests**: Universal properties using FsCheck.NET
- **Integration Tests**: End-to-end scenarios where applicable

### Test Structure

```
tests/Nivara.Tests/
├── Unit/                    # Unit tests for specific functionality
├── Properties/              # Property-based tests for correctness
├── Integration/             # Integration tests
└── Performance/             # Performance and benchmark tests
```

### Running Tests

```bash
# Run all tests
dotnet test

# Run specific test project
dotnet test tests/Nivara.Tests

# Run with coverage
dotnet test --collect:"XPlat Code Coverage"

# Run property-based tests with verbose output
dotnet test --logger "console;verbosity=detailed"
```

## 📝 Coding Standards

### C# Conventions

- **Private fields**: Use simple `Type fieldName` (no underscore prefix)
- **Local variables**: Use camelCase
- **Public properties**: Use PascalCase
- **Method organization**: Static public → public instance → private methods
- **Local functions**: Keep at the top of containing method

### Code Quality

- Use meaningful variable and method names
- Add XML documentation for public APIs
- Follow SOLID principles
- Minimize memory allocations
- Prefer `Span<T>` and `Memory<T>` for performance-critical code
- Use `unsafe` code only when necessary for performance

### Example Class Structure

```csharp
public sealed class ExampleClass : IDisposable
{
    // Static public methods first
    public static ExampleClass Create() => new();
    
    // Public properties and methods
    public int Length { get; }
    
    public void DoSomething()
    {
        // Local functions at top
        bool isValid(int value) => value > 0;
        
        // Implementation
    }
    
    // Private fields (no underscore)
    readonly int[] data;
    bool disposed;
    
    // Private methods (camelCase)
    void cleanup()
    {
        // Implementation
    }
}
```

## 🔧 Performance Guidelines

### Memory Management

- Use memory pools for frequently allocated objects
- Prefer `stackalloc` for small, short-lived arrays
- Implement `IDisposable` for types that manage resources
- Use `Span<T>` and `Memory<T>` for zero-copy operations

### SIMD Optimization

- Use `System.Numerics.Vector<T>` for vectorized operations
- Ensure data is properly aligned for SIMD operations
- Provide scalar fallbacks for non-SIMD hardware
- Test performance on different architectures

### Benchmarking

- Use BenchmarkDotNet for performance measurements
- Include benchmarks for performance-critical code
- Test on multiple platforms (Windows, Linux, macOS)
- Document performance characteristics

## 📚 Documentation

### XML Documentation

All public APIs must have XML documentation:

```csharp
/// <summary>
/// Creates a new NivaraSeries with the specified data.
/// </summary>
/// <param name="data">The data to store in the series.</param>
/// <param name="validityMask">Optional validity mask for null handling.</param>
/// <returns>A new NivaraSeries instance.</returns>
public static NivaraSeries<T> Create(ReadOnlySpan<T> data, ReadOnlySpan<bool> validityMask = default)
```

### Code Comments

- Explain complex algorithms and optimizations
- Document performance considerations
- Explain non-obvious design decisions
- Use TODO comments for future improvements

## 🐛 Bug Reports

### Before Reporting

1. Check existing issues for duplicates
2. Verify the bug with the latest version
3. Create a minimal reproduction case

### Bug Report Template

```markdown
**Description**
A clear description of the bug.

**Reproduction Steps**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen.

**Actual Behavior**  
What actually happens.

**Environment**
- OS: [Windows/Linux/macOS]
- .NET Version: [10.0/11.0/etc.]
- Nivara Version: [version]

**Additional Context**
Any other relevant information.
```

## 💡 Feature Requests

### Before Requesting

1. Check the [roadmap](README.md#-roadmap) and existing issues
2. Consider if the feature aligns with project goals

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature.

**Use Case**
Why is this feature needed? What problem does it solve?

**Proposed API**
Example of how the feature would be used.

**Alternatives Considered**
Other approaches you've considered.

**Additional Context**
Any other relevant information.
```

## 🔄 Pull Request Process

### Before Submitting

1. Ensure all tests pass
2. Update documentation as needed
3. Add appropriate tests for new functionality
4. Follow the coding standards
5. Update the task status if implementing a planned task

### PR Template

```markdown
**Description**
Brief description of changes.

**Related Issue/Task**
Link to related issue or task from the task list.

**Type of Change**
- [ ] Bug fix
- [ ] New feature  
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement

**Testing**
- [ ] Unit tests added/updated
- [ ] Property-based tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass

**Checklist**
- [ ] Code follows project coding standards
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process

1. **Automated Checks**: CI builds and tests must pass
2. **Code Review**: At least one maintainer review required
3. **Testing**: Verify tests cover new functionality
4. **Documentation**: Ensure documentation is updated
5. **Merge**: Squash and merge after approval

## 🏷️ Versioning

Nivara follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## 📞 Getting Help

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: Check the docs/ directory for guides

## 🙏 Recognition

Contributors will be recognized in:

- Release notes for their contributions
- Contributors section in documentation
- GitHub contributors list

Thank you for contributing to Nivara! 🚀