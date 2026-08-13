# TODO — #226: `_`-prefixed private fields in core files

## Problem

`OptimizationRule.cs` and `QueryPlanVisitor.cs` use `_`-prefixed private fields,
violating AGENTS.md ("Private fields: camelCase without `_` prefix") and the
`.editorconfig` `private_fields_should_be_camel_case` rule (severity suggestion,
`required_prefix` empty, `capitalization` camel_case).

## Proposed change

Rename fields to camelCase and update all in-file references:

1. `src/Nivara/Optimization/OptimizationRule.cs` (`OptimizationEngine`)
   - `_rules` → `rules`
2. `src/Nivara/Query/QueryPlanVisitor.cs` (`QueryPlanStatisticsVisitor`)
   - `_operationCounts` → `operationCounts` (also `_`-prefixed, same rule)
   - `_totalOperations` → `totalOperations`
   - `_maxDepth` → `maxDepth`
   - `_currentDepth` → `currentDepth`
3. `src/Nivara/Query/QueryPlanVisitor.cs` (`QueryPlanValidationVisitor`)
   - `_errors` → `errors`
   - `_currentSchema` → `currentSchema`

## Ctor shadowing

`OptimizationEngine(IEnumerable<OptimizationRule> rules)` assigns to the field —
after rename, field `rules` shadows ctor param `rules`. Use
`this.rules = rules?.ToList() ?? throw new ArgumentNullException(nameof(rules));`.
No ctor shadowing in the visitor classes (they have no constructors).

## Blast radius

Fields are `private`; all references are file-local. External consumers use the
public properties (`Rules`, `OperationCounts`, `TotalOperations`, `MaxDepth`,
`Errors`, `IsValid`) and constructors — unaffected. Tests referencing
`OptimizationEngine` and `QueryPlanStatisticsVisitor` exercise public API only.

## Verification

- `dotnet build Nivara.slnx`
- `dotnet test` (ask human first) — Query/Execution suites cover the touched
  classes (`QueryExecutorTests`, `ExecutionEngineTests`, `WindowPlanVisitorTests`).

## Commits

1. `docs: plan #226 in TODO.md`
2. `refactor: rename _-prefixed private fields to camelCase (#226)`
3. `docs: remove TODO.md — plan executed` (after confirming nothing pending)

## GitHub issues log

- None yet. As work executes, any deferred work / concern discovered must be
  created immediately via `gh issue create --repo khurram-uworx/Nivara` and the
  number recorded here — don't rely on memory.
