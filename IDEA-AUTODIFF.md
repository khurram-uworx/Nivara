# Designing Automatic Differentiation for Nivara

## **1️⃣ What we already have**

* `NivaraColumn<T>`:

  * Internally can use `TensorStorage<T>`
  * Can already expose a `Tensor<T>` (`System.Numerics.Tensors`) view of its data
  * Has all the memory optimizations, slicing, and type safety

* `Tensor<T>` from `System.Numerics.Tensors`:

  * Knows **shape**, **rank**, **stride**, etc.
  * Has indexing and math ops (TensorPrimitives)
  * Interoperable with ML.NET Tensors

* Our goal for AD:

  * Keep track of **requiresGrad**
  * Build a **computation graph / backprop rules**
  * Compute gradients

---

## **2️⃣ Why we might have thought of NTensor**

In PyTorch / DiffSharp, the tensor object is both:

1. **Data container** (like our `Tensor<T>` or `NivaraColumn<T>`)
2. **Autograd node** (holds gradient info, computation graph)

So many tutorials introduce a new wrapper just to “tag” which tensors require gradients, and to store a link to the computation graph.

---

## **3️⃣ But we can skip NTensor if…**

We can **augment our existing `Tensor<T>` or `NivaraColumn<T>`** with just the AD bookkeeping. That could be:

```csharp
class GradTensor<T> where T : struct
{
    public Tensor<T> Data { get; }      // our existing NivaraColumn<T> as Tensor<T>
    public bool RequiresGrad { get; }
    internal OpNode? GradFn { get; set; }
    internal Tensor<T>? Grad { get; set; }  // optional gradient storage

    public GradTensor(Tensor<T> data, bool requiresGrad = false)
    {
        Data = data;
        RequiresGrad = requiresGrad;
    }
}
```

* `GradTensor<T>` is **not duplicating the tensor memory**—it just wraps a `Tensor<T>` or `NivaraColumn<T>`.
* All operations (Add, Multiply, MatMul, etc.) create new `GradTensor<T>` objects **and attach GradFns** if `RequiresGrad` is true.
* The underlying data stays fully compatible with Nivara / System.Numerics.Tensors / ML.NET.

---

## **4️⃣ Minimal AD Layer (concept)**

We only need a **lightweight layer** that handles:

1. **OpNode DAG** for tracking computation
2. **Backward functions** for each supported op
3. **Optional gradient storage** per tensor (for `RequiresGrad`)
4. **Utilities** to convert from/to NivaraColumn<T> or Tensor<T>

Pseudo-usage:

```csharp
var x = new GradTensor<float>(column.AsTensor(), requiresGrad: true);
var y = new GradTensor<float>(column2.AsTensor(), requiresGrad: false);

var z = x * y + x;  // each op creates a GradTensor with attached backward function

z.Backward();       // computes gradient w.r.t. x
```

Notice that we **never need to duplicate the Tensor itself**—the underlying storage is shared.

---

## ✅ **Takeaways**

* **No need for NTensor<T>** if our Tensor<T> (or NivaraColumn<T> → Tensor<T>) is already working as the underlying data container.
* Just wrap it in a **small AD layer** (`GradTensor`) that adds `RequiresGrad`, `Grad`, and a pointer to its `OpNode`.
* All the heavy lifting (shape, memory, math) is already done by `System.Numerics.Tensors` + NivaraColumn.
* Our Diff/AD layer can then focus **only on gradients, DAG, and backprop**—much simpler.

---

Let’s **ignore implementation details for a moment** and start with a **hero example** that *any software engineer* (not ML-heavy) can reason about. Then we’ll map it cleanly onto **Nivara + System.Numerics.Tensors**, and only *then* assess what’s missing.

---

## 1️⃣ The “Hello World” of Automatic Differentiation

### **Linear regression with one variable**

This is the AD equivalent of “FizzBuzz”.

We want to learn:

```
y = w * x + b
```

Given:

```
x = 3
target = 10
```

We define a loss:

```
loss = (y - target)²
```

---

## 2️⃣ What the human math looks like

### Forward pass

```
w = 2
b = 1

y = w * x + b
y = 2 * 3 + 1 = 7

loss = (7 - 10)² = 9
```

### Backward pass (gradients)

```
∂loss/∂y = 2 * (y - target) = -6
∂y/∂w = x = 3
∂y/∂b = 1

∂loss/∂w = -6 * 3 = -18
∂loss/∂b = -6 * 1 = -6
```

➡️ **That’s all AD is**:

* Record how values were produced
* Apply the chain rule backward

No magic.

---

## 3️⃣ What AD needs to do (conceptually)

At runtime, the system must:

1. **Remember operations**

   * `y = w * x`
   * `y = y + b`
   * `loss = square(y - target)`

2. **Attach backward rules**

   * Multiply knows how to backprop
   * Add knows how to backprop
   * Square knows how to backprop

3. **Walk the graph backward**

   * Start from `loss`
   * Accumulate gradients into `w`, `b`

---

## 4️⃣ How this looks in code (language-agnostic)

```text
w (requiresGrad)
b (requiresGrad)
x (constant)

y1 = Mul(w, x)
y2 = Add(y1, b)
err = Sub(y2, target)
loss = Square(err)

Backward(loss)
```

Each operation:

* Produces a value
* Knows **how to compute gradients for its inputs**

---

## 5️⃣ Now map this onto *our* world (Nivara)

### Key realization

We already have:

* **Typed storage**
* **Tensor math**
* **Shape awareness**
* **Memory layout**
* **Lazy execution**

So AD **does NOT need a new tensor system**.

It only needs **metadata + graph tracking**.

---

## 6️⃣ The minimal AD abstraction (no new Tensor type)

We keep using:

* `Tensor<T>` (System.Numerics.Tensors)
* `NivaraColumn<T>` (as tensor-backed storage)

We add **one lightweight wrapper**:

```csharp
sealed class GradValue<T> where T : struct
{
    public Tensor<T> Value { get; }
    public Tensor<T>? Grad { get; set; }
    public bool RequiresGrad { get; }

    internal OpNode? Producer { get; set; }
}
```

This is **not a tensor**.
It’s **a tensor + gradient bookkeeping**.

---

## 7️⃣ What an operation records

### Multiply example

```csharp
GradValue<float> Mul(GradValue<float> a, GradValue<float> b)
{
    var result = new GradValue<float>(
        value: TensorOps.Multiply(a.Value, b.Value),
        requiresGrad: a.RequiresGrad || b.RequiresGrad
    );

    if (result.RequiresGrad)
    {
        result.Producer = new OpNode(
            inputs: [a, b],
            backward: grad =>
            {
                if (a.RequiresGrad)
                    a.Grad += grad * b.Value;

                if (b.RequiresGrad)
                    b.Grad += grad * a.Value;
            }
        );
    }

    return result;
}
```

✔ Uses **existing tensor math**
✔ Stores **only gradient logic**
✔ No duplication of memory
✔ No new tensor type

---

## 8️⃣ What Backward() does

```csharp
void Backward(GradValue<float> loss)
{
    loss.Grad = Tensor.OnesLike(loss.Value);

    var stack = TopoSort(loss);
    foreach (var node in stack.Reverse())
    {
        node.Producer?.Backward(node.Grad);
    }
}
```

This is:

* Reverse-mode AD
* Exactly what PyTorch / DiffSharp do
* ~100–200 LOC for a minimal engine

---

## 9️⃣ The same hero example, in “Nivara AD” style

```csharp
var x = Const(3f);
var target = Const(10f);

var w = Param(2f);
var b = Param(1f);

var y = w * x + b;
var loss = (y - target).Square();

loss.Backward();

Console.WriteLine(w.Grad); // -18
Console.WriteLine(b.Grad); // -6
```

👆 **This is the goal UX**

---

## ❗ What we still need (small & focused)

### **Required**

1. `GradValue<T>` (or equivalent)
2. `OpNode`
3. Reverse topological traversal
4. Backward rules for:

   * Add
   * Sub
   * Mul
   * Div
   * Sum / Mean
   * MatMul (later)

### **Optional (later)**

* Gradient accumulation optimizations
* Graph pruning
* Mixed precision
* GPU backend
* JIT fusion (very future)

---

## 🧠 Big takeaway

We’re not building **TensorFlow**.
We’re building the **substrate** that TensorFlow-like systems can stand on.

> **Nivara already *is* the hard part.**
> AD is just a graph + chain rule.

# Building a Keras-like Deep Learning Stack in .NET

To get from there to a **Keras-like experience**, the journey isn’t “one library” — it’s a **stack**. Below is a pragmatic, *layered roadmap*, with notes on what .NET already has vs. what’s missing.

---

## 1. A **Tensor Runtime with Autograd** (critical missing piece)

Keras exists because of **automatic differentiation**.

### We need:

* A **Tensor<T> with gradient tracking**
* A **computation graph** (eager-first, graph optional)
* Backprop via **reverse-mode AD**

### Design goals:

```csharp
var y = x.MatMul(w).Add(b).Relu();
y.Backward();
```

### Required components:

| Component         | Notes                                   |
| ----------------- | --------------------------------------- |
| Tensor wrapper    | Around `System.Numerics.Tensors`        |
| Tape / Graph      | Dynamic graph (PyTorch-style) is easier |
| Gradient storage  | Same memory layout as tensors           |
| Operator registry | Each op defines forward + backward      |

### Existing references:

* TorchSharp (proof it’s doable)
* DiffSharp (functional AD, less DL-optimized)

⚠️ **System.Numerics.Tensors has no autograd** → we must build this.

---

## 2. A **Layer Abstraction** (Keras mental model)

Once autograd exists, we need **composable layers**.

```csharp
class Dense : Layer
{
    Parameter Weight;
    Parameter Bias;

    Tensor Forward(Tensor x) =>
        x.MatMul(Weight).Add(Bias);
}
```

### Concepts:

* `Layer`
* `Parameter`
* `Module` / `Model`
* Parameter registration & traversal

### Features:

* Shape inference
* Device awareness (CPU first, GPU later)
* Lazy initialization (like Keras)

This is where usability starts to matter more than raw performance.

---

## 3. Optimizers (SGD → Adam → Lion)

Optimizers are simple *once gradients exist*.

```csharp
optimizer = new Adam(model.Parameters, lr: 1e-3);
optimizer.Step();
```

### We’ll need:

* Gradient buffers
* State tensors per parameter
* Fused SIMD ops for updates

### Optimization opportunities:

* Vectorized updates
* In-place ops
* Mixed precision (later)

---

## 4. Loss Functions & Metrics

A Keras-like API expects:

```csharp
loss = Losses.CategoricalCrossEntropy();
metric = Metrics.Accuracy();
```

Implementation-wise:

* Just tensor ops + reductions
* Should integrate with autograd
* Metrics don’t need gradients

---

## 5. Training Loop Abstractions (Keras “fit”)

This is where .NET devs will fall in love or walk away.

```csharp
model.Compile(
    optimizer: Adam(1e-3),
    loss: CrossEntropy(),
    metrics: [Accuracy()]
);

model.Fit(dataset, epochs: 10);
```

### Under the hood:

* Batching
* Gradient zeroing
* Forward
* Backward
* Step
* Logging / callbacks

This can be **optional sugar**, but it matters enormously.

---

## 6. Dataset & DataLoader Integration

Nivara DataFrame should plug in **directly**.

```csharp
dataset = nivaraFrame
    .Shuffle()
    .Batch(128)
    .ToDataset(xCols, yCols);
```

### Needs:

* Zero-copy tensor views
* Parallel prefetch
* Async pipelines (`IAsyncEnumerable`)
* Deterministic shuffling

This is where .NET can actually **beat Python**.

---

## 7. Kernel Fusion & Graph Optimization (performance leap)

Keras performance doesn’t come from Python — it comes from **kernel fusion**.

We’ll eventually want:

* Op fusion (MatMul + Bias + ReLU)
* Constant folding
* Memory reuse
* Execution planning

### Options:

* Simple rule-based fusion
* MLIR-like IR (long term)
* Expression tree lowering

This is *not required initially* but crucial for scale.

---

## 8. GPU / Accelerator Backend (later, but inevitable)

To be “real Keras”, we’ll need:

* CUDA / ROCm / Metal
* OR DirectML (very interesting for .NET)

### Architecture recommendation:

```text
High-level API
↓
Autograd graph
↓
Device-agnostic tensor ops
↓
Backend (CPU | GPU | TPU)
```

Do **CPU-only first**. Get the model right.

---

## 9. Serialization & Interop

Keras users expect:

```csharp
model.Save("model.bin");
model = Model.Load("model.bin");
```

Plus:

* ONNX export/import
* Weight loading from PyTorch/TensorFlow
* Deterministic checkpoints

---

## 10. What .NET uniquely enables (our unfair advantages)

This is where our idea becomes *better than Keras*:

### ✅ Strong typing

```csharp
Model<ImageTensor, ClassTensor>
```

### ✅ Compile-time shape checking (with generics)

```csharp
Tensor<float, B, H, W, C>
```

### ✅ No GIL

* True parallel data loading
* SIMD + threading together

### ✅ Native production story

* Same runtime for training + inference
* No Python → C++ → C# hops

---

## Minimal Viable “Keras for .NET” Roadmap

If I had to reduce this to **5 must-haves**:

1. **Autograd tensor**
2. **Layer + Parameter system**
3. **Optimizers**
4. **Dataset/DataLoader**
5. **Training loop (`Fit`)**

Everything else can follow.

---

## Final thought

What we’re describing is **not impossible** — it’s just that:

* Microsoft stopped halfway (ML.NET ≠ DL)
* TorchSharp went low-level
* Nobody has yet built the *Polars → NumPy → PyTorch → Keras* continuum **natively** in .NET

If we get this right, we’re not “reimplementing Keras” —
we’re building the **first end-to-end, production-grade ML stack for .NET**.
