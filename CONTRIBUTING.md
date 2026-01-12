# Contributing to Nivara

Thank you for your interest in contributing to **Nivara**, a high-performance, columnar DataFrame library for .NET. This document defines **how humans contribute** to the project: workflow, structure, standards, and review expectations.

> **Important**
> - This file is **human-owned**.
> - AI tools may **suggest** changes, but should not directly modify this file.
> - Architectural rationale, gotchas, and LLM-specific learnings belong in **GUIDELINES.md**, not here.
> - Feature descriptions and user-facing documentation belong in **README.md**, not here.

---

## 🚀 Getting Started

### Prerequisites

- .NET 10.0 SDK or later
- Git
- A code editor (Visual Studio, VS Code, or JetBrains Rider recommended)

### Repository Setup

Clone the repository and restore dependencies:

```bash
git clone https://github.com/khurram-uworx/nivara.git
cd nivara
dotnet restore
```

### Build

```bash
dotnet build
```

### Run Tests

```bash
dotnet test
```

---

## 🗂️ Project Structure

Understanding the repository layout is essential before making changes.

### Core Library

```
src/Nivara/
├── Diagnostics/               # Performance analysis and diagnostic tools
├── Exceptions/                # Custom exception hierarchy
├── Execution/                 # Execution strategies and engine
├── Expressions/               # Query expression system
├── Helpers/                   # Utility classes and helpers
├── IO/                        # Built-in IO (core data sources)
├── Operations/                # DataFrame operations (joins, aggregations, etc.)
├── Optimization/              # Query optimization rules
├── Query/                     # Query engine and planning
├── Storage/                   # Column storage implementations
├── Tensors/                   # Tensor-based operations and interop
├── Schema.cs                  # Schema management
├── NivaraColumn.cs            # Column implementation
├── NivaraSeries.cs            # Series implementation
├── NivaraFrame.cs             # DataFrame implementation
├── NivaraFrameExtensions.cs   # DataFrame extension methods
├── IColumn.cs                 # Column interface
├── IColumnStorage.cs          # Column storage interface
└── IFrame.cs                  # DataFrame interface
```

### Extensions

Third-party dependencies and optional integrations live in a separate project:

```
src/Nivara.Extensions/
├── IO/
│   ├── ArrowInterop.cs      # Apache Arrow integration
│   ├── ParquetReader.cs     # Parquet file support
│   ├── ParquetWriter.cs     # Parquet file writing
│   ├── CsvDataSource.cs     # CSV support (CsvHelper)
│   ├── CsvExtensions.cs     # CSV factory methods
│   └── [Other IO Files]     # Buffer management, streaming, type mapping
└── MLNet/
    ├── MLNetExtensions.cs   # ML.NET integration
    ├── MLNetInterop.cs      # ML.NET interoperability
    ├── ModelIntegration.cs  # Model integration helpers
    └── TensorConversions.cs # Tensor conversion utilities
```

### Tests

Tests mirror the source structure for discoverability:

```
tests/Nivara.Tests/
├── Diagnostics/           # Diagnostic system tests
├── Exceptions/            # Exception handling tests
├── Execution/             # Execution strategy tests
├── Expressions/           # Expression system tests
├── Helpers/               # Helper utility tests
├── IO/                    # Core I/O tests
├── MLNet/                 # ML.NET integration tests
├── Operations/            # DataFrame operations tests
├── Optimization/          # Query optimization tests
├── Query/                 # Query engine tests
├── Storage/               # Storage implementation tests
├── Tensors/               # Tensor operations tests
├── SchemaTests.cs         # Schema validation tests
├── NivaraColumnTests.cs   # Column implementation tests
├── NivaraSeriesTests.cs   # Series implementation tests
├── NivaraFrameTests.cs    # DataFrame core tests
├── IntegrationTests.cs    # End-to-end integration tests
└── [Property-based Tests] # Property tests
```

---

## 🔄 Development Workflow

1. **Choose a Task**
   - Pick an issue or roadmap item that aligns with project goals.

2. **Create a Branch**
   - Use a descriptive feature or fix branch name.

3. **Implement Changes**
   - Follow coding standards and existing architectural patterns.
   - Keep changes focused and incremental.

4. **Add Tests**
   - All new behavior must be covered by appropriate tests.

5. **Update Documentation**
   - User-facing changes → update **README.md**
   - Implementation patterns and gotchas → update **CHANGELOGS.md**
   - Transferable engineering lessons → consider **GUIDELINES.md** (but avoid project-specific details)
   - Do *not* add architectural rationale to this file

6. **Submit a Pull Request**
   - Provide a clear description of what changed and why.

---

## 🧪 Testing Guidelines

All contributions must include tests where applicable.

### Test Types

- **Unit Tests**: Concrete examples and edge cases (NUnit)
- **Property-Based Tests**: Invariants and universal properties
- **Integration Tests**: End-to-end scenarios and complex workflows
- **Performance Tests**: Benchmarks for performance-critical paths (BenchmarkDotNet)

### Testing Best Practices

- Test each operation with multiple data types (int, double, string, nullable types)
- Explicitly test null handling scenarios
- Test empty inputs and single-element inputs
- Test error conditions with descriptive assertions
- Group related tests in nested test classes for organization
- Use property-based tests to validate invariants across wide input spaces
- Do not rely solely on example-based tests for core logic
- Test both immediate execution (DataFrame) and lazy execution (QueryFrame) paths
- Validate query optimization behavior and execution strategies
- Test interoperability with external formats (Arrow, Parquet, CSV) in Extensions

### Running Tests

```bash
# All tests
dotnet test

# With coverage
dotnet test --collect:"XPlat Code Coverage"

# Run specific test categories
dotnet test --filter "Category=Unit"
dotnet test --filter "Category=Integration"
dotnet test --filter "Category=Property"

# Run performance benchmarks (if available)
dotnet run --project benchmarks --configuration Release
```

---

## 📝 Coding Standards

### C# Conventions

- Public APIs use **PascalCase**
- Local variables use **camelCase**
- Private fields use simple names (no underscore prefix)
- Static public methods first, then instance methods, then private methods
- Add XML documentation to all public APIs

### Code Quality Expectations

- Prefer clarity over cleverness
- Avoid unnecessary allocations
- Follow SOLID principles
- Use `Span<T>` / `Memory<T>` where appropriate
- Use `unsafe` code only when clearly justified and documented

### Architectural Principles

- **Correctness Over Performance**: Never apply optimizations unless semantic safety can be proven
- **Explicit Over Implicit**: Make execution boundaries, error conditions, and type behavior explicit
- **Conservative Optimization**: If optimization safety is uncertain, skip the optimization
- **Null Handling**: Use explicit boolean masks for null tracking, never sentinel values
- **Immutable Data Model**: Operations return new data structures rather than modifying existing ones
- **Schema-First Design**: Validate schemas early to catch errors before expensive operations
- **Lazy vs Eager Execution**: Provide both immediate (DataFrame) and deferred (QueryFrame) execution models

### Type System Guidelines

- Prefer runtime type inspection over complex generic constraints
- Use type-erased interfaces for runtime operations beneath generic APIs
- Handle nullable types explicitly using `Nullable.GetUnderlyingType()`
- Avoid reflection-heavy designs for core execution paths
- Use vectorization only when measurably beneficial and fallback to scalar operations

### Query System Guidelines

- Implement operations as `IQueryOperation` for integration with query planning
- Use conservative optimization strategies that preserve correctness
- Support multiple execution strategies (lazy, eager, parallel, streaming)
- Provide comprehensive diagnostics and performance analysis capabilities
- Validate schemas during `TransformSchema` phase to fail fast

---

## 📦 Performance and Benchmarks

- Use **BenchmarkDotNet** for performance measurements
- Include benchmarks for performance-sensitive code
- Validate behavior on multiple platforms when possible
- Test both vectorized and scalar code paths
- Benchmark different execution strategies (lazy, eager, parallel, streaming)
- Profile memory usage and allocation patterns
- Compare performance against baseline implementations

---

## 🔧 Development Tools and Setup

### Recommended Extensions (VS Code)

- C# Dev Kit
- .NET Install Tool
- GitLens
- Test Explorer UI

### Recommended Extensions (Visual Studio)

- BenchmarkDotNet templates
- Code coverage tools

### Build Configuration

The project uses:
- **.NET 10.0** as the target framework
- **C# latest** language version
- **Nullable reference types** enabled
- **Unsafe code blocks** allowed for performance-critical paths
- **System.Numerics.Tensors** for vectorization

### Package Structure

- **Nivara** (core): Zero external dependencies except System.Numerics.Tensors
- **Nivara.Extensions**: Optional integrations with third-party libraries (CsvHelper, Apache Arrow, Parquet.Net, ML.NET)

---

## 🔁 Pull Request Process

### Before Submitting

- All tests pass (unit, integration, and property-based tests)
- Documentation updated if behavior is user-visible
- Code follows project standards and architectural principles
- Changes are limited to the intended scope
- Performance impact has been considered and measured if relevant
- Null handling follows explicit boolean mask patterns
- New operations integrate with the query system where appropriate

### Review Requirements

- CI must be green
- At least one maintainer review
- Clear commit history (squash merges preferred)

---

## 🏷️ Versioning

Nivara follows **Semantic Versioning**:

- **MAJOR** – breaking changes
- **MINOR** – new features (backward compatible)
- **PATCH** – bug fixes

---

## 💬 Getting Help

- **GitHub Issues** – bugs and feature requests
- **GitHub Discussions** – questions and design discussions
- **CHANGELOGS.md** – implementation patterns, gotchas, and architectural decisions
- **GUIDELINES.md** – transferable engineering lessons for similar systems

---

## 🎯 Contribution Areas

We welcome contributions in these areas:

### Core Library
- Performance optimizations (with benchmarks)
- New DataFrame operations
- Query optimization rules
- Execution strategy improvements
- Better error messages and diagnostics

### Extensions
- New file format support
- Enhanced ML.NET integration
- Additional data source connectors
- Streaming and large dataset optimizations

### Testing & Quality
- Property-based test coverage
- Performance regression tests
- Cross-platform compatibility testing
- Memory usage optimization

### Documentation
- API documentation improvements
- Usage examples and tutorials
- Performance guides
- Migration guides from other DataFrame libraries

---

Thank you for contributing to Nivara. Your help makes the project better 🚀
