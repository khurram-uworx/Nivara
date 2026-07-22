---
name: iterative-commit
description: Use when completing discrete steps of multi-step work. Commits locally after each logical change unit, does NOT push. Push is always controlled by the human.
---

# Iterative Commit Workflow

When working on multi-step tasks, commit after each logical change unit so the human can review incremental progress.

## Rules

1. **Commit frequently** — after completing each discrete step (bug fix, feature addition, refactor, test update, etc.)
2. **Never push** — commits are local only. The human pushes manually after reviewing changes outside the session.
3. **Write clear commit messages** — describe what changed and why, in imperative mood.
4. **Stage selectively** — only stage files relevant to the completed step, not unrelated changes.
5. **Verify before committing** — run lint/typecheck/build/tests if available before each commit.
6. **One logical change per commit** — don't bundle unrelated changes together.

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

1. Complete the step (code change + build/test verification)
2. `git status` to see changed files
3. `git diff` to review changes
4. `git add <specific files>` — stage only the files for this step
5. `git commit -m "<message>"`
6. Report to the human what was committed (without pushing)
7. Continue to next step

## What NOT to do

- Do NOT use `git push` at any point
- Do NOT amend previous commits unless explicitly asked
- Do NOT use interactive rebase or squash
- Do NOT commit secrets, keys, or credentials
- Do NOT commit generated files or build artifacts
