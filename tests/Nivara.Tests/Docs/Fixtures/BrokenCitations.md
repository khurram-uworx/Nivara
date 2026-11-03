# Negative control for the documentation citation gate

This is not reader-facing documentation. It is a fixture under `tests/`, which is outside the
gate's scan scope (repository root plus `docs/**`), so the repository-wide run never sees it.
The citation gate reads this file deliberately and is asserted to reject **every** citation
below. If any of them starts passing, the gate has stopped proving anything and a green run
means nothing.

## Files that do not exist

A qualified path to nowhere: `src/Nivara/Storage/NoSuchFileDoesNotExist.cs::Foo`

A bare basename matching nothing in the repository: `NoSuchFileAnywhere.cs::Foo`

## A basename that exists in more than one file

`Program.cs::Foo`

## Symbols that do not exist

`src/Nivara/Execution/NivaraExecutionContext.cs::DefinitelyNotAMember`

## Symbols with ambiguous overloads and no signature

`src/Nivara/Schema.cs::GetHashCode`

## Citations that must still pass, so the fixture proves it is not simply rejecting everything

`src/Nivara/Execution/NivaraExecutionContext.cs::MemoryBudget`
