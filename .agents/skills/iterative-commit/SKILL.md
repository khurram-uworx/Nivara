---
name: iterative-commit
description: Use when completing discrete steps of multi-step work. Suggests/asks for a feature branch and switches to it, writes the plan to docs/TODO.md first, commits locally after each logical change unit (does NOT push), asks before running tests, deletes docs/TODO.md once the plan is fully executed, then offers to push and create/update a PR. Push and PRs are always human-confirmed.
---

# Iterative Commit Workflow

When working on multi-step tasks, commit after each logical change unit so the human can review incremental progress.

## Rules

1. **Commit frequently** — after completing each discrete step (bug fix, feature addition, refactor, test update, etc.)
2. **Never push** — commits are local only. The human pushes manually after reviewing changes outside the session.
3. **Write clear commit messages** — describe what changed and why, in imperative mood.
4. **Stage selectively** — only stage files relevant to the completed step, not unrelated changes.
5. **Verify before committing** — run lint/typecheck/build/tests if available before each commit.
6. **Ask before running tests** — `dotnet test` and other long-running verification commands require explicit human confirmation before starting (see AGENTS.md).
7. **One logical change per commit** — don't bundle unrelated changes together.

## Plan-first workflow

Persist the plan before executing so it is saved at highest fidelity, even if context is later lost.

1. **Suggest/ask for the branch** — propose a short feature branch name (e.g., `khurram/<feature>`), ask the human to confirm, then create it off the current base (typically `main`): `git checkout -b <branch>`. Do not proceed until the human confirms the branch.
2. **Write the plan to `docs/TODO.md` first** — document the problem, proposed changes (with code sketches where useful), verification steps, planned commit list, and follow-ups. Commit it as its own logical unit (`docs: plan <work> in TODO.md`).
3. **Execute iteratively** — complete one logical change at a time, committing after each (see Workflow below). Ask before running `dotnet test`.
4. **Review `docs/TODO.md` when the plan is complete** — read it over and confirm every item is taken care of. If so, remove it and commit the removal (`git rm docs/TODO.md` → `docs: remove TODO.md — plan executed`). Only leave it in place if an item is still pending.
5. **Offer push + PR** — report the completed work, then offer to push the branch and create (or update) a pull request. Ask explicitly; do not push or open a PR without the human's confirmation. Push remains human-controlled by default.

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
5. `git add <specific files>` — stage only the files for this step
6. `git commit -m "<message>"`
7. Report to the human what was committed (without pushing)
8. Continue to next step

## What NOT to do

- Do NOT use `git push` at any point
- Do NOT amend previous commits unless explicitly asked
- Do NOT use interactive rebase or squash
- Do NOT commit secrets, keys, or credentials
- Do NOT commit generated files or build artifacts
