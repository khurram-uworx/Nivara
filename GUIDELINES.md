# GUIDELINES

## Purpose

This document captures **transferable engineering lessons** discovered while building data-intensive, lazy-execution systems — and, in the last major section, while establishing whether such systems actually work. It is intentionally **not project-specific** and avoids recording historical implementation details.

Its primary audience is:
- Future humans working on similar systems
- AI agents assisting with design, refactoring, and optimization

The goal is to **prevent repeated exploration of known dead ends** and to encode patterns that reliably worked under real constraints.

> Focus on **why certain approaches failed or succeeded**, not on *what* was implemented in a specific codebase.

The most expensive mistakes are not in code that is wrong. They are in code that is wrong **and reported as passing**, and in conclusions drawn from results that were real but narrower than claimed. That is why [Verification, Evidence & Claim Discipline](#verification-evidence--claim-discipline) is a section in its own right rather than a footnote.

---

## How to Use This Document

Each section is written using the same structure:

- **Problem** – A recurring engineering challenge
- **Constraint** – Why the obvious solution fails
- **Pattern That Worked** – A reusable solution
- **Negative Rule** – What to avoid next time
- **Outcome** – Why this pattern generalizes

When extending this document:
- Avoid project names and concrete APIs
- Prefer abstraction over narration
- Capture failure modes aggressively

---

## Core Engineering Principles

### Correctness Over Performance

**Problem**  
Performance optimizations often introduce subtle semantic bugs.

**Constraint**  
Many optimizations are only conditionally safe and require deep dependency analysis.

**Pattern That Worked**  
Adopt a correctness-first mindset: a slow but correct system is preferable to a fast incorrect one.

**Negative Rule**  
Never apply an optimization unless its semantic safety can be proven.

**Outcome**  
This dramatically reduces hard-to-debug correctness issues and builds trust in the system.

---

### Explicit Over Implicit Behavior

**Problem**  
Implicit behavior (auto-coercion, hidden execution, silent fallbacks) obscures system behavior.

**Constraint**  
Implicit systems are hard to reason about, test, and optimize.

**Pattern That Worked**  
Make execution boundaries, error conditions, and type behavior explicit.

**Negative Rule**  
Avoid "magic" behavior that changes execution strategy without visibility.

**Outcome**  
Systems become easier to debug, explain, and extend.

---

## Data & Null Semantics

### Avoid NaN-Based Null Semantics

**Problem**  
Representing missing data in numeric systems is non-trivial.

**Constraint**  
NaN-based null semantics:
- Do not work for integer types
- Leak into user-visible results
- Behave inconsistently across operations

**Pattern That Worked**  
Track nulls explicitly using boolean masks that participate in all operations.

**Negative Rule**  
Do not encode nulls using sentinel numeric values.

**Outcome**  
Null behavior becomes predictable, type-agnostic, and testable.

---

### Null Propagation Rules

**Problem**  
Operations involving null values often produce ambiguous results.

**Constraint**  
Implicit null handling leads to silent data corruption.

**Pattern That Worked**  
Define a strict rule: *any operation involving a null produces a null*.

**Negative Rule**  
Never "fill" or ignore nulls implicitly during computation.

**Outcome**  
Data integrity is preserved across all transformations.

---

## Generic & Type System Design

### Type Erasure for Runtime Systems

**Problem**  
Query engines and expression evaluators often need to operate on values of unknown generic types at runtime.

**Constraint**  
Purely generic designs break down when types must be stored, inspected, or dispatched dynamically.

**Pattern That Worked**  
Introduce a type-erased interface for runtime operations, layered beneath a generic, type-safe API.

**Negative Rule**  
Avoid reflection-heavy designs for core execution paths.

**Outcome**  
Runtime flexibility is achieved without sacrificing compile-time safety for user code.

---

### Over-Constraining Generics

**Problem**  
Strong generic constraints appear attractive for numeric and algebraic systems.

**Constraint**  
Compile-time generic constraints quickly become unmanageable for:
- Operator overloading
- Mixed-type operations
- Extensible execution engines

**Pattern That Worked**  
Prefer runtime type inspection and dispatch in complex generic scenarios.

**Negative Rule**  
Do not attempt to encode all numeric rules into generic constraints.

**Outcome**  
The system remains flexible and evolvable.

---

## Lazy Execution & Deferred Errors

### Deferred Error Handling Trade-offs

**Problem**  
Lazy systems defer work — and therefore errors — until execution.

**Constraint**  
Deferring errors can:
- Mask failures during query construction
- Cause cascading validation issues later

**Pattern That Worked**  
Defer only *unavoidable* errors while validating structure and schema eagerly.

**Negative Rule**  
Do not defer errors that invalidate the logical correctness of a query.

**Outcome**  
Lazy semantics are preserved without sacrificing debuggability.

---

## Query Planning & Optimization

### Conservative Optimization Strategy

**Problem**  
Query reordering and optimization can dramatically improve performance.

**Constraint**  
Many optimizations subtly change semantics when dependencies exist.

**Pattern That Worked**  
Apply only conservative optimizations unless full dependency analysis proves safety.

**Negative Rule**  
If optimization safety is uncertain, skip the optimization.

**Outcome**  
Correctness is preserved while still allowing meaningful performance gains.

---

### Dependency Awareness

**Problem**  
Optimizations often assume independence between operations.

**Constraint**  
Expressions may reference overlapping data, making reordering unsafe.

**Pattern That Worked**  
Analyze dependencies explicitly before transforming execution plans.

**Negative Rule**  
Never reorder operations blindly.

**Outcome**  
Optimization remains predictable and correct.

---

## Interoperability & External APIs

### Zero-Copy Aspirations vs Reality

**Problem**  
High-performance systems often promise zero-copy operations for interoperability.

**Constraint**  
True zero-copy requires:
- Shared memory layout assumptions
- Compatible lifetime management
- Exposed internal data structures

**Pattern That Worked**  
Design for zero-copy but implement with copying initially. Optimize to true zero-copy only when:
- Performance profiling proves it's necessary
- Internal APIs can safely expose underlying data
- Memory layout compatibility is guaranteed

**Negative Rule**  
Do not compromise internal API design for theoretical zero-copy benefits.

**Outcome**  
Systems remain flexible and correct while preserving optimization opportunities.

---

### Method Overload Resolution in Generic Contexts

**Problem**  
Generic methods with similar signatures create ambiguous overload resolution.

**Constraint**  
Type inference often fails when:
- Multiple generic methods match the same call pattern
- Optional parameters create multiple valid resolutions
- Return types differ but parameters are identical

**Pattern That Worked**  
Design method signatures to be unambiguous:
- Use different parameter counts or types
- Provide explicit disambiguation parameters
- Consider separate method names for fundamentally different operations

**Negative Rule**  
Do not rely on return type differences alone to distinguish overloads.

**Outcome**  
API calls become predictable and less error-prone.

---

### Make Performance Explainable

**Problem**  
Users struggle to trust systems that optimize invisibly.

**Constraint**  
Silent optimizations make debugging and tuning difficult.

**Pattern That Worked**  
Expose diagnostic information explaining:
- Execution strategy
- Optimization decisions
- Performance characteristics

**Negative Rule**  
Do not hide execution behavior from users.

**Outcome**  
Users gain insight and confidence without needing internal knowledge.

---

## Aggregation & Grouping Operations

### Composite Key Design for Grouping

**Problem**  
Grouping operations need to handle multiple columns with different types and null values efficiently.

**Constraint**  
Simple string concatenation for composite keys:
- Fails with null values
- Creates ambiguous keys (e.g., "AB" + "C" vs "A" + "BC")
- Poor performance for numeric types
- Inconsistent hash distribution

**Pattern That Worked**  
Create a dedicated composite key class with proper equality, hashing, and null handling:
- Use object arrays to preserve type information
- Implement proper GetHashCode using HashCode.Add for each component
- Handle nulls explicitly in equality comparisons
- Provide meaningful ToString for debugging

**Negative Rule**  
Never use string concatenation or simple tuple hashing for composite keys in production grouping operations.

**Outcome**  
Grouping becomes reliable, performant, and debuggable across all data types.

---

### Nullable Type Handling in Generic Operations

**Problem**  
Runtime type dispatch fails when dealing with nullable value types (e.g., `int?` vs `int`).

**Constraint**  
Type switch expressions don't automatically handle nullable variants:
- `typeof(int?)` doesn't match `typeof(int)` patterns
- Aggregation functions need to work on both nullable and non-nullable columns
- Type validation becomes complex with nullable generics

**Pattern That Worked**  
Use `Nullable.GetUnderlyingType()` to normalize types before dispatch:
- Extract underlying type for nullable types, use original type for non-nullable
- Apply type-specific logic to the underlying type
- Handle null extraction at the value level, not the type level

**Negative Rule**  
Do not create separate code paths for nullable and non-nullable variants of the same underlying type.

**Outcome**  
Single code path handles both nullable and non-nullable types correctly, reducing complexity and maintenance burden.

---

### Vectorization Strategy for Aggregations

**Problem**  
Aggregation functions need to balance performance with correctness across diverse data types.

**Constraint**  
Vectorization libraries only support specific types and require contiguous memory:
- Not all numeric types are vectorizable
- Null values break vectorization assumptions
- Mixed-type operations complicate vectorization

**Pattern That Worked**  
Implement a fallback hierarchy:
1. Use vectorized operations for supported types when no nulls present
2. Fall back to scalar operations for unsupported types or when nulls exist
3. Extract valid values first, then apply vectorized operations to clean data
4. Always provide scalar implementations as the baseline

**Negative Rule**  
Never assume vectorization is always faster - measure and provide fallbacks.

**Outcome**  
Aggregations are both fast (when vectorizable) and correct (always), with predictable behavior across all scenarios.

---

### ArrayPool Buffers Must Be Cleared on Rent

**Problem**  
`ArrayPool<T>.Shared.Rent(size)` returns buffers that may contain stale data from previous usage. When these buffers are used as accumulators or intermediate results, stale values produce silent numerical corruption — most commonly NaN propagation in floating-point paths.

**Constraint**  
- The pool contract does not guarantee zeroed memory
- Clearing every rented buffer on every call adds overhead
- The bug is invisible until the stale value happens to be nonzero

**Pattern That Worked**  
Clear rented buffers immediately after allocation (`buf.AsSpan(0, size).Clear()`) in setup/initialization paths where the buffer will be reused across iterations. For per-call hot paths, clear only the portion of the buffer that will be read before it is fully written.

**Negative Rule**  
Never assume `ArrayPool` returns zeroed memory. Always clear when the buffer is used as an accumulator or when stale values could affect correctness.

**Outcome**  
Eliminates a class of silent numerical bugs that are extremely difficult to diagnose because they depend on allocation reuse order.

---

### Property-Based Testing

**Problem**  
Traditional unit tests fail to cover combinatorial edge cases.

**Constraint**  
Complex systems exhibit failures only under unexpected input combinations.

**Pattern That Worked**  
Use property-based tests to validate invariants across wide input spaces.

**Negative Rule**  
Do not rely solely on example-based tests for core logic.

**Outcome**  
Subtle correctness issues are caught early.

Property-like tests can be implemented with **parameterized test suites (NUnit)** rather than full FsCheck because:

- FsCheck has a strong reputation in the F# community but is less visible in mainstream C# circles
- FsCheck though is respected but is not cultural fit for C# and there are not many examples out there
- LLMs might not have seen using FsCheck in C# and will struggle to suggest correct code
- AI Agents therefore struggle when using FsCheck in C# codebase and we loose lot of AI credits
- Therefore AI Agents should stay away from FsCheck

---

### Testing Strategy for Complex Operations

**Problem**  
Complex operations like grouping and aggregation have many edge cases and type combinations that are difficult to test comprehensively.

**Constraint**  
Manual test case enumeration:
- Misses edge cases with null values
- Doesn't cover all type combinations
- Becomes unmaintainable as operations grow
- Fails to catch subtle correctness issues

**Pattern That Worked**  
Combine comprehensive unit testing with systematic edge case coverage:
- Test each operation with multiple data types (int, double, string, nullable types)
- Explicitly test null handling scenarios
- Test empty inputs and single-element inputs
- Test error conditions with descriptive assertions
- Group related tests in nested test classes for organization

**Negative Rule**  
Do not rely on "happy path" testing alone for complex data operations.

**Outcome**  
High confidence in correctness across all supported scenarios, with clear test organization that makes maintenance easier.

---

## Verification, Evidence & Claim Discipline

The sections above are about systems. This one is about the evidence used to decide whether a
system works, which is where the expensive mistakes live: not in the code that is wrong, but in
the code that is wrong *and reported as passing*, and in the conclusions drawn from results
that were real but narrower than claimed.

### Scope Every Claim to the Data That Survives It

**Problem**
A result measured across many cases gets compressed into one sentence, written from the cases
that agreed.

**Constraint**
*Consistently*, *always*, *every* and *never* are quantified universals, and they get checked
against the average rather than the worst row. The average hides the rows that inverted. A claim
holding on 9 of 15 cases and reversing on 3 is not a claim with a caveat — it is two claims, and
the second usually carries the real mechanism, because the regime boundary is the interesting
finding.

**Pattern That Worked**
Compute the claim against every row of the table it summarizes before publishing it, not a
representative subset. When the effect turns out to be regime-dependent, promote the regime into
the claim: *"the effect dominates when the device is saturated and does not predict at all when
it is not."* That is more useful than the false universal, because it tells a reader whether the
result applies to their case.

**Negative Rule**
Do not write *consistently* unless you checked it against the row that broke it. If a mechanism
is conditional, publish the condition.

**Outcome**
Readers can apply the finding to their own situation, and the next person does not re-run the
experiment to discover where the claim stops holding.

---

### A Gate Must State Its Own Coverage

**Problem**
A verification step contributes no cases — nothing loads, everything is filtered out, the set
under test comes out empty — and the summary it prints is unchanged.

**Constraint**
Coverage loss is silent. Pass/fail is computed over whatever actually ran, so a gate that tested
three quarters of what it claims reports a quarter's result with the same confidence. The
failure mode is not a wrong number passing; it is a correct number answering a smaller question
than the one being asked.

**Pattern That Worked**
Make the summary state coverage every time — *N of M loaded*, with a reason per omission — and
make reduced coverage an explicit failure rather than a neutral event. Keep "cannot run here" and
"ran and passed" distinguishable: a unit the current device or environment cannot host is
reported as skipped with the reason, never silently dropped.

**Negative Rule**
Never let a summary read identically before and after a coverage change. If new work contributes
nothing, the gate should say so and fail.

**Outcome**
A green gate always means the same amount of testing as last time, and a summary cannot be
mistaken for a smaller result.

---

### One Failure Reason Per Catch, Per Counter, Per Message

**Problem**
One mechanism absorbs several unrelated failure modes: a single `catch` around a block that can
fail for two distinct reasons, or a single counter behind two different verdicts.

**Constraint**
The second reason is always the more serious one, and it is absorbed without comment. A catch
justified by *"the environment might not support this"* also catches *"this does not compile"*,
which is a defect in the code, and reports it as an environment limitation. One counter behind
two verdicts produces a message that is wrong at exactly the moment it is read, because the
counts get summed and the summary then claims a numeric bound was exceeded when the real failure
was structural.

**Pattern That Worked**
Classify at the point of detection and keep the classes separate all the way to the exit code:
distinct catch blocks for distinct reasons, a distinct counter per verdict, and a message per
verdict naming the verdict it belongs to. Aggregate only at the end, by summing counts and never
by merging labels.

**Negative Rule**
Do not widen a catch or a counter to cover a neighbouring failure. If a second reason can occur
in that block, give it its own path.

**Outcome**
The reported failure always matches the actual failure, and *"the environment could not run
this"* can never be confused with *"this is broken"*.

---

### Assert Exactness When Exactness Is Available

**Problem**
A new implementation is required to reproduce a reference — same operation order, same rounding —
and is gated on a numeric tolerance instead.

**Constraint**
A tolerance converts a structural defect into a judgement call, and arguing with a judgement
call is the path of least resistance, so tolerance gates drift to whatever the current result
happens to be. An exactness assertion cannot be argued with: either the ordering changed or it
did not, and the failure names the cells involved. It also catches classes of bug a tolerance
cannot see — a mis-placed staging buffer returning the right order of magnitude against the
wrong elements, or a compilation failure that a skipped case would have hidden entirely.

**Pattern That Worked**
When the contract is "reproduce this reference exactly", assert that first and keep the
tolerance as a second, independent assertion rather than a replacement. Note in the gate why
the exactness check exists, so a future reader does not "simplify" it away.

**Negative Rule**
Do not substitute a tolerance for an exactness requirement the implementation is designed to
meet — and equally, do not bolt exactness onto code that genuinely cannot provide it. Assert it
where the design promises it.

**Outcome**
Structural defects surface on the first case rather than after a tolerance argument, and the
check doubles as a design-regression detector.

---

### One Commit, One Reason

**Problem**
An unrelated change rides along in a commit whose subject is about something else — a rename, a
formatting pass, a duplicate removed while tidying nearby lines.

**Constraint**
The reader of the fix is not looking for it, so it is never reviewed, and it does not appear in
the message because it was not intentional. If it touches a test's input set it also invalidates
the historical baselines the next person compares against, and they will compare against numbers
that no longer describe the code.

**Pattern That Worked**
Keep the commit to its stated reason. When a drive-by change is genuinely correct, land it
separately with its own message, so the fix stays reviewable on its own terms and the baseline
change is visible as a change.

**Negative Rule**
Do not let a commit alter the inputs, thresholds, or counts of a test it is not about. If it
does, it belongs in its own commit.

**Outcome**
Every commit is reviewable against its own stated reason, and a historical number always
describes a code state that existed.

---

### Superseded Measurements Move Together

**Problem**
A measurement is corrected, and the correction is applied to the places the old value happens
to appear in the files currently open.

**Constraint**
Partially updated figures are worse than uniformly stale ones, because they are internally
inconsistent in a way that reads as intentional. A share re-quoted next to its complement now
sums past 100%, and nobody notices, because each number is individually plausible and the two
live in different files.

**Pattern That Worked**
When a measurement is superseded, find every site carrying any part of it and move them all in
one change — including the numbers in a partition, which are separate claims about the same
run. Then check the arithmetic: parts of a partition should sum to the whole, and two figures
describing the same measurement should agree. Those two free checks catch nearly every partial
propagation.

**Negative Rule**
Do not update a measurement only at the sites you happen to be editing, and never leave a
corrected value next to an uncorrected one from the same run.

**Outcome**
No document states two values for the same measurement, and a partition that does not sum to its
total fails loudly.

---

### Measure the Share in the Denominator You Will Optimize

**Problem**
A component's share of total *work* is used to argue that it dominates total *cost*.

**Constraint**
The two denominators disagree, sometimes by an order of magnitude. Work share is dominated by
whichever term has the most operations; cost share is dominated by whichever term is
latency-bound, launch-bound, or cache-bound. A term can be a rounding error in work and the
single largest cost in wall-clock — and computing the first number precisely only makes you
more confident about optimizing the wrong thing.

**Pattern That Worked**
Time each component in isolation, at the configuration it is really launched with, and multiply
by its actual per-forward count. Do this *before* designing the optimization, not to validate it
afterwards: a probe whose job includes being able to say "the premise is wrong" is worth ten
times more before the work than after. Attribute to components rather than buckets — "the
matrix multiply" hides more than it explains, and splitting it is what surfaces the real term.

**Negative Rule**
Do not scope a performance effort from a share of work, a share of operations, or a share of
parameters. Measure time.

**Outcome**
Effort goes to the term that actually costs the time, and a proposal whose premise fails
measurement dies in a day rather than after a kernel family is built.

---

### Verify the Process You Care About

**Problem**
A verification step passes its output filter, and the exit status of the process under test is
never observed.

**Constraint**
A pipeline reports the last stage's status, so a filtered command reports the *filter's* status
— always zero, when the filter matched. And a check that only inspects output text cannot
distinguish "ran and passed" from "never ran", because the most common way a check silently
stops testing anything produces no output at all. A check written by whoever wrote the thing it
checks shares that person's assumptions, which is exactly where the defect lives.

**Pattern That Worked**
Capture the exit status from the process under test — not from anything downstream of it — and
assert on it separately from the output. Where a check can go vacuous, break it deliberately and
confirm it fails: a gate never observed failing is not known to work. Have the review done by
someone who did not write the code.

**Negative Rule**
Do not treat a printed success message as a passing exit status. Do not accept a verification
step you have not seen fail. Do not rely solely on your own review of your own assumptions.

**Outcome**
Green means the process succeeded, red means it failed, and the checks are known to tell the
two apart because they have been seen doing it.

---

## What Didn't Work (High-Value Failures)

### Reflection-Heavy Designs

- Fragile
- Hard to optimize
- Difficult to reason about

### Premature Vectorization

- Increases complexity
- Often provides marginal gains
- Obscures correctness issues

### Over-Specification

- Encoding too many rules upfront reduces flexibility
- Systems evolve faster than rigid designs

---

## AI-Specific Guidance

### Where Tokens Are Commonly Wasted

- Re-deriving known null-handling strategies
- Over-engineering generic constraints
- Attempting unsafe query optimizations

### Preferred Defaults for AI Agents

- Start with correctness
- Choose explicit designs
- Avoid cleverness unless justified
- Measure before optimizing, and be willing to report that the premise was wrong
  ([Measure the Share in the Denominator You Will Optimize](#measure-the-share-in-the-denominator-you-will-optimize))
- Assert the exit status of the process under test, not of a filter in a pipeline, and never
  report a check you have not seen fail
  ([Verify the Process You Care About](#verify-the-process-you-care-about))
- Scope a claim to the rows that survive it, and say when a mechanism is regime-dependent
  ([Scope Every Claim to the Data That Survives It](#scope-every-claim-to-the-data-that-survives-it))
- Do not merge an unrelated change into a fix commit
  ([One Commit, One Reason](#one-commit-one-reason))
- When superseded figures appear in several files, update all of them in one change, then check
  the arithmetic
  ([Superseded Measurements Move Together](#superseded-measurements-move-together))

### Self-Review Is Not Verification

An agent reviewing its own work is checking its output against its own assumptions, so it
structurally cannot find a defect that comes from a shared assumption. The two most common
instances: a gate reports success while testing less than it claims, and a claim is stated more
broadly than the data supports — both invisible to the author because both follow from
assumptions the author also holds. When a check can pass vacuously, break it deliberately and
confirm it fails. When a result is load-bearing for a decision, have someone who did not write
it re-derive the conclusion from the raw data.

---

## Closing Note

This document is intentionally **opinionated and distilled**. It should evolve slowly and only when a *new, broadly applicable lesson* is learned.

If a lesson only applies to a specific implementation, it does **not** belong here.
