---
name: iterative-work
description: Use when completing discrete steps of multi-step work. Suggests/asks for a feature branch and switches to it, writes the plan to docs/TODO.md first, commits locally after each logical change unit (does NOT push), asks before running tests, deletes docs/TODO.md once the plan is fully executed, then offers to push and create/update a PR. Push and PRs are always human-confirmed.
---

# Iterative Work Workflow

When working on multi-step tasks, commit after each logical change unit so the human can review incremental progress.

## Rules

1. **Commit frequently** — after completing each discrete step (bug fix, feature addition, refactor, test update, etc.)
2. **Never push** — commits are local only. The human pushes manually after reviewing changes outside the session.
3. **Write clear commit messages** — describe what changed and why, in imperative mood.
4. **Stage selectively** — only stage files relevant to the completed step, not unrelated changes.
5. **Verify before committing** — run lint/typecheck/build/tests if available before each commit.
6. **Ask before running tests** — `dotnet test` and other long-running verification commands require explicit human confirmation before starting (see AGENTS.md).
7. **One logical change per commit** — don't bundle unrelated changes together.
8. **Create GitHub issues during execution, not after** — as each task executes, if you find deferred work or a concern (known limitations, follow-ups, refactors) that is outside the current plan, create a tracked issue immediately via `gh issue create --repo khurram-uworx/Nivara` and record its number in the GitHub issues log in `docs/TODO.md`. Don't rely on memory or wait until the plan finishes — compaction can lose it.

## Plan-first workflow

Persist the plan before executing so it is saved at highest fidelity, even if context is later lost.

1. **Suggest/ask for the branch** — propose a short feature branch name (e.g., `khurram/<feature>`), ask the human to confirm, then create it off the current base (typically `main`): `git checkout -b <branch>`. Do not proceed until the human confirms the branch.
2. **Write the plan to `docs/TODO.md` first** — document the problem, proposed changes (with code sketches where useful), verification steps, planned commit list, and a **GitHub issues log** (see below). Include the reminder: as each task executes, if you find deferred work or a concern, create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its number in the log — don't rely on memory or wait until the end of the plan, as compaction during execution can lose important items. Commit it as its own logical unit (`docs: plan <work> in TODO.md`).
3. **Execute iteratively** — complete one logical change at a time, committing after each (see Workflow below). Ask before running `dotnet test`.
4. **Review `docs/TODO.md` when the plan is complete** — read it over and confirm every item is taken care of. If so, remove it and commit the removal (`git rm docs/TODO.md` → `docs: remove TODO.md — plan executed`). Only leave it in place if an item is still pending.
5. **Offer push + PR** — report the completed work, then offer to push the branch and create (or update) a pull request. Ask explicitly; do not push or open a PR without the human's confirmation. Push remains human-controlled by default.

## GitHub issues log

Include this section in every `docs/TODO.md` plan and keep it updated as work progresses:

```
## GitHub issues log

- [ ] #NNN — one-line description (created while working on <task>)
```

- Create the issue at discovery time via `gh issue create --repo khurram-uworx/Nivara`, then record the issue number here.
- If the issue already exists, reference its number instead of creating a duplicate.
- The log also feeds the final review — confirm each captured issue is still relevant before removing `docs/TODO.md`.

## Grounding the plan

- **Tensors/Vectors are new in .NET** — always ground the implementation plan and strategy in official documentation and examples fetched via the microsoft-learn MCP server (`microsoft_docs_search`, `microsoft_code_sample_search`, `microsoft_docs_fetch`) before committing to an approach.
- **Navigate source with code-memory MCP** — when exploring the codebase for planning or design, prefer the code-memory MCP tools (symbols, relationships, call graphs, impact analysis, you will love its sql tool) over ad-hoc grepping.
- **Assess blast radius** — for every plan, identify and state the blast radius: what files/components/symbols each change affects, what depends on them (downstream callers), and which tests cover them. Include this in `docs/TODO.md` so the human can judge risk before execution.

## Commit Message Format

```
<short summary in imperative mood>

<optional body explaining why and what changed>
```

Examples:
```
Fix NaN in Adam optimizer from uninitialized ArrayPool buffers

ArrayPool.Rent() does not zero buffers. First step used garbage
values for expAvg/expAvgSq, producing NaN in weight updates.
Added AsSpan(0,size).Clear() after every Rent in all optimizers.
```

```
Add oProj output projection to TransformerBlock forward pass

MultiHeadAttention result was bypassing the output projection,
going directly to the residual add. oProj was allocated but
never called, resulting in null gradients for the parameter.
```

## Workflow

1. Complete the step (code change + build verification)
2. Ask the human before running `dotnet test` or any long-running verification command
3. `git status` to see changed files
4. `git diff` to review changes
5. Capture deferred work — scan the step for anything worth doing later (limitations, follow-ups, concerns). If found, create a GitHub issue now (`gh issue create --repo khurram-uworx/Nivara`) and record it in the GitHub issues log in `docs/TODO.md`; never hold it in memory.
6. `git add <specific files>` — stage only the files for this step
7. `git commit -m "<message>"`
8. Report to the human what was committed (without pushing)
9. Continue to next step

## When a test fails

A failing test is an opportunity to assess the root cause properly — not a signal to patch around it. Before changing anything:

1. **Diagnose first** — determine whose expectations are wrong:
   - Did **our change** introduce a regression (a bug in the code)?
   - Did the **design/contract** change (feature, API, or behavior intentionally changed, so the old expectation no longer holds)?
   - Is the **test stale** — asserting behavior we superseded?
2. **If our code change is the problem**, fix the code — don't edit the test to make it pass.
3. **If the expectation legitimately changed** (design change, new contract), update the test to assert the new correct behavior and record the intent in the commit message / `docs/TODO.md`.
4. **Don't rush quick fixes** — approach it the right engineering way, even if it takes longer.

## Test coverage balance

- **Existing tests are the consistency guardrail** — as we progress and sometimes break things, the existing suite is how we catch regressions and stay consistent. When adding something new, keep existing tests green unless a contract deliberately changed (see "When a test fails").
- **Cover new work with good-enough tests** — enough to verify the new behavior and its edge cases, but not so much that tests become brittle or a maintenance chore. Prefer extending the existing suite over adding redundant new tests.
- **Keep tests focused and readable** — assert behavior, not implementation details; over-testing internals makes refactors painful.

## What NOT to do

- Do NOT use `git push` at any point
- Do NOT amend previous commits unless explicitly asked
- Do NOT use interactive rebase or squash
- Do NOT commit secrets, keys, or credentials
- Do NOT commit generated files or build artifacts
