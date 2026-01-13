# Nivara.Extensions.AutoDiff

This directory contains the automatic differentiation (AD) implementation for Nivara.Extensions.

## Directory Structure

```
AutoDiff/
├── README.md                           # This file
├── IGradOperation.cs                   # Interface for gradient-aware operations
├── IAutoGradNumeric.cs                 # Interface for numeric types supporting AD
├── GradTensor.cs                       # Main AD tensor wrapper (forward declaration)
├── OpNode.cs                          # Computation graph node (forward declaration)
├── ComputationGraph.cs                # Graph management (forward declaration)
├── Exceptions/
│   └── AutoGradExceptions.cs          # AD exception hierarchy (forward declarations)
├── Extensions/
│   └── NivaraAutoGradExtensions.cs    # Nivara integration extensions (forward declaration)
├── Operations/
│   └── GradOperations.cs              # Gradient-aware operations (forward declaration)
└── Utilities/
    └── GradientUtils.cs               # Gradient utilities (forward declaration)
```

## Implementation Status

This is the initial project structure setup (Task 1). All files contain forward declarations and will be fully implemented in subsequent tasks:

- **Task 2**: Core GradTensor infrastructure
- **Task 3**: Gradient-aware operation wrappers  
- **Task 4**: Backward pass and gradient computation
- **Task 6**: Advanced operation wrappers
- **Task 7**: Null handling and Nivara integration
- **Task 8**: Gradient utilities and memory management
- **Task 9**: Comprehensive error handling
- **Task 10**: Type safety and generic constraints
- **Task 11**: Nivara integration extensions

## Core Interfaces

### IGradOperation<T>
Defines the contract for operations that can compute both forward values and gradients.

### IAutoGradNumeric<T>
Defines the requirements for numeric types that support automatic differentiation.

## Integration with Nivara

The AD system is designed to integrate seamlessly with existing Nivara types:
- **NivaraColumn<T>**: Primary data storage for tensors
- **NivaraSeries<T>**: Series integration for gradient computation
- **NivaraFrame**: Batch operations on multiple columns
- **Existing Storage Backends**: Leverages TensorStorage and MemoryStorage

## Namespace Organization

All automatic differentiation functionality is contained within the `Nivara.Extensions.AutoDiff` namespace and its sub-namespaces:
- `Nivara.Extensions.AutoDiff.Operations`
- `Nivara.Extensions.AutoDiff.Utilities`
- `Nivara.Extensions.AutoDiff.Exceptions`
- `Nivara.Extensions.AutoDiff.Extensions`