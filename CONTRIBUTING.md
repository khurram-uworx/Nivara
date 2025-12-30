# Contributing to Nivara DataFrame Library

Thank you for your interest in contributing to Nivara! This document provides guidelines and information for contributors.

---

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

---

## 📦 Project Structure
```bash
src/
├── Nivara/                 # Core library (dependency-free)
│   ├── Diagnostics/        # Performance analysis and diagnostic tools
│   ├── Exceptions/         # Custom exception hierarchy
│   ├── Expressions/        # Query expression system
│   ├── IO/                 # Built-in IO functionality (JSON, etc.)
│   ├── Memory/             # Memory-based storage implementations
│   ├── Tensors/            # Tensor-based storage implementations
│   └── [Root Files]        # Core interfaces and main classes
├── Nivara.Extensions/      # Extension methods and third-party integrations
│   └── IO/                 # Third-party IO functionality (CSV, Parquet, etc.)
samples/
├── Nivara.SampleApp/       # Sample applications
tests/
├── Nivara.Tests/           # Unit and property tests
│   ├── Diagnostics/        # Tests for diagnostic functionality
│   ├── Memory/             # Tests for memory storage
│   ├── Tensors/            # Tests for tensor storage
│   └── [Root]              # Tests for core functionality
```

---

## 📋 Development Process

### Finding Work
- Review the [project status](#project-status) in the README.
- Look for issues labeled `good first issue` or `help wanted`.

### Implementation Workflow
1. **Choose a Task**: Select an available task from the task list.
2. **Review Specifications**: Read the relevant requirements and design sections.
3. **Create Branch**: Create a feature branch for your work.
4. **Implement**: Follow the coding standards and implement the feature.
5. **Test**: Add comprehensive unit and property-based tests.
6. **Document**: Update documentation and examples as needed.
7. **Submit PR**: Create a pull request with a clear description.

---

## 🧪 Testing Guidelines

### Test Requirements
All contributions must include appropriate tests:
- **Unit Tests**: Specific examples and edge cases using NUnit.
- **Property-Based Tests**: Universal properties using FsCheck.NET.
- **Integration Tests**: End-to-end scenarios where applicable.

### Test Structure
```bash
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

---

## 📝 Coding Standards

### C# Conventions
- **Private fields**: Use simple `Type fieldName` (no underscore prefix).
- **Local variables**: Use camelCase.
- **Public properties**: Use PascalCase.
- **Method organization**: Static public → public instance → private methods.
- **Local functions**: Keep at the top of containing method.

### Code Quality
- Use meaningful variable and method names.
- Add XML documentation for public APIs.
- Follow SOLID principles.
- Minimize memory allocations.

---

## 🔧 Performance Guidelines

### Memory Management
- Use memory pools for frequently allocated objects.
- Prefer `stackalloc` for small, short-lived arrays.
- Implement `IDisposable` for types that manage resources.
- Use `Span<T>` and `Memory<T>` for zero-copy operations.

### SIMD Optimization
- Use `System.Numerics.Vector<T>` for vectorized operations.
- Ensure data is properly aligned for SIMD operations.
- Provide scalar fallbacks for non-SIMD hardware.
- Test performance on different architectures.

### Benchmarking
- Use BenchmarkDotNet for performance measurements.
- Include benchmarks for performance-critical code.
- Test on multiple platforms (Windows, Linux, macOS).
- Document performance characteristics.

---

## 🔄 Pull Request Process

### Before Submitting
1. Ensure all tests pass.
2. Update documentation as needed.
3. Add appropriate tests for new functionality.
4. Follow the coding standards.
5. Update the task status if implementing a planned task.

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
```

---

## 📞 Getting Help
- **Discussions**: Use GitHub Discussions for questions.
- **Issues**: Report bugs and request features via GitHub Issues.
- **Documentation**: Check the docs/ directory for guides.

---

## 🙏 Recognition
Contributors will be recognized in:
- Release notes for their contributions.
- Contributors section in documentation.
- GitHub contributors list.

Thank you for contributing to Nivara! 🚀
