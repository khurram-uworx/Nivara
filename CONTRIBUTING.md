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
├── Diagnostics/           # Performance analysis and diagnostic tools
├── Exceptions/            # Custom exception hierarchy
├── Expressions/           # Query expression system
├── IO/                    # Built-in IO (JSON, core data sources)
├── Memory/                # Memory-based storage (non-vectorizable types)
├── Tensors/               # Tensor-based storage (vectorizable types)
├── Schema.cs              # Schema management
├── NivaraColumn.cs        # Column implementation
├── NivaraSeries.cs        # Series implementation
├── NivaraFrame.cs         # DataFrame implementation
└── [Query Engine Files]   # Query planning and execution
```

### Extensions

Third-party dependencies and optional integrations live in a separate project:

```
src/Nivara.Extensions/
└── IO/
    ├── CsvDataSource.cs   # CSV support (CsvHelper)
    └── CsvExtensions.cs   # CSV factory methods
```

### Tests

Tests mirror the source structure for discoverability:

```
tests/Nivara.Tests/
├── Diagnostics/
├── Expressions/
├── Memory/
├── Tensors/
├── SchemaTests.cs
├── NivaraColumnTests.cs
├── NivaraSeriesTests.cs
└── NivaraFrameTests.cs
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
   - Do *not* add rationale or lessons learned here (use GUIDELINES.md instead).

6. **Submit a Pull Request**
   - Provide a clear description of what changed and why.

---

## 🧪 Testing Guidelines

All contributions must include tests where applicable.

### Test Types

- **Unit Tests**: Concrete examples and edge cases (NUnit)
- **Property-Based Tests**: Invariants and universal properties (FsCheck.NET)
- **Integration Tests**: End-to-end scenarios where relevant
- **Performance Tests**: Benchmarks for performance-critical paths

### Testing Best Practices

- Test each operation with multiple data types (int, double, string, nullable types)
- Explicitly test null handling scenarios
- Test empty inputs and single-element inputs
- Test error conditions with descriptive assertions
- Group related tests in nested test classes for organization
- Use property-based tests to validate invariants across wide input spaces
- Do not rely solely on example-based tests for core logic

### Running Tests

```bash
# All tests
dotnet test

# With coverage
dotnet test --collect:"XPlat Code Coverage"
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

### Type System Guidelines

- Prefer runtime type inspection over complex generic constraints
- Use type-erased interfaces for runtime operations beneath generic APIs
- Handle nullable types explicitly using `Nullable.GetUnderlyingType()`
- Avoid reflection-heavy designs for core execution paths

---

## 📦 Performance and Benchmarks

- Use **BenchmarkDotNet** for performance measurements
- Include benchmarks for performance-sensitive code
- Validate behavior on multiple platforms when possible

---

## 🔁 Pull Request Process

### Before Submitting

- All tests pass
- Documentation updated if behavior is user-visible
- Code follows project standards
- Changes are limited to the intended scope

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

- GitHub Issues – bugs and feature requests
- GitHub Discussions – questions and design discussions

---

Thank you for contributing to Nivara. Your help makes the project better 🚀
