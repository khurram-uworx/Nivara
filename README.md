# Nivara
**A high-performance, type-safe DataFrame library for .NET**

[![Build Status](https://img.shields.io/github/actions/workflow/status/khurram-uworx/nivara/ci.yml?branch=main)](https://github.com/khurram-uworx/nivara/actions)
[![NuGet](https://img.shields.io/nuget/v/Nivara.svg)](https://www.nuget.org/packages/Nivara)
[![License](https://img.shields.io/github/license/khurram-uworx/nivara)](LICENSE)

---

## 📢 Contributing & Roadmap
- 👉 **Want to contribute?** See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, guidelines, and workflows.
- 👉 **Project Roadmap**: [GitHub Projects](https://github.com/khurram-uworx/nivara/projects)

---

## 🚀 Quick Start

### Install via NuGet
```bash
dotnet add package Nivara
```

### Create Your First DataFrame
```csharp
using Nivara;
using Nivara.Columns;

// Create a DataFrame with typed columns
var df = new DataFrame(
    new Int32Column("Age", new[] { 25, 30, 35 }),
    new StringColumn("Name", new[] { "Alice", "Bob", "Charlie" })
);

// Query with LINQ-like syntax
var adults = df.Filter(row => row.Get<int>("Age") >= 30);
adults.PrettyPrint();
```
**Output:**
```
| Age | Name    |
|-----|---------|
| 30  | Bob     |
| 35  | Charlie |
```

---

## 🔥 Features

### 1. **Type-Safe, Immutable Columns**
- Strongly-typed columns for compile-time safety.
- Immutable by design for thread safety and predictability.

```csharp
var ages = new Int32Column("Age", new[] { 25, 30, 35 });
var names = new StringColumn("Name", new[] { "Alice", "Bob", "Charlie" });
var df = new DataFrame(ages, names);
```

### 2. **Automatic Storage Optimization**
- **Vectorizable types** (int, float, double) use SIMD-optimized tensor storage.
- **Non-vectorizable types** (string, Guid) use memory-efficient storage.

```csharp
// Automatically uses tensor storage for ints
var intCol = new Int32Column("Values", Enumerable.Range(0, 1000).ToArray());
// Uses memory storage for strings
var strCol = new StringColumn("Labels", Enumerable.Range(0, 1000).Select(i => $"Label{i}").ToArray());
```

### 3. **Vectorized Operations**
- SIMD-accelerated arithmetic and logical operations.

```csharp
var a = new Int32Column("A", new[] { 1, 2, 3 });
var b = new Int32Column("B", new[] { 4, 5, 6 });
var sum = a + b; // Vectorized addition
```

### 4. **Comprehensive Query Engine**
- Lazy-evaluated, composable queries.

```csharp
var filtered = df
    .Filter(row => row.Get<int>("Age") > 25)
    .Select(row => new { Name = row.Get<string>("Name"), AgeNextYear = row.Get<int>("Age") + 1 });
```

### 5. **Schema Management**
- Runtime schema validation and evolution.

```csharp
var schema = new Schema(
    new ColumnSchema("Age", typeof(int), isNullable: false),
    new ColumnSchema("Name", typeof(string), isNullable: true)
);
var validatedDf = df.Cast(schema);
```

### 6. **Built-in I/O**
- JSON, CSV, and Parquet support.

```csharp
// Save to CSV
df.WriteCsv("data.csv");
// Load from JSON
var loadedDf = DataFrame.ReadJson("data.json");
```

### 7. **Performance Diagnostics**
- Built-in tools for profiling and optimization.

```csharp
var stats = df.Profile();
Console.WriteLine(stats);
```

---

## 📊 Example: Data Analysis Workflow

```csharp
// Load data
var df = DataFrame.ReadCsv("sales.csv");

// Filter and transform
var highValueSales = df
    .Filter(row => row.Get<decimal>("Amount") > 1000)
    .Select(row => new {
        Product = row.Get<string>("Product"),
        Amount = row.Get<decimal>("Amount") * 1.1m // Apply 10% tax
    });

// Group and aggregate
var salesByProduct = highValueSales
    .GroupBy(row => row.Product)
    .Select(g => new {
        Product = g.Key,
        Total = g.Sum(row => row.Amount)
    });

// Output results
salesByProduct.PrettyPrint();
```

---

## 📦 Available Packages

| Package                     | Description                                     |
|-----------------------------|-------------------------------------------------|
| `Nivara`                    | Core DataFrame library                          |
| `Nivara.Extensions`         | CSV and Parquet I/O support, ML.NET Integration |

---

## 🤝 Community
- **Discussions**: [GitHub Discussions](https://github.com/khurram-uworx/nivara/discussions)
- **Issues**: [GitHub Issues](https://github.com/khurram-uworx/nivara/issues)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📜 License
Nivara is [MIT licensed](LICENSE).
