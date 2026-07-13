---
name: verification-before-completion
description: Enforce evidence-based completion claims. Use before claiming ANY task is done, before any commit/push/PR, and before moving to the next task. Claiming work is complete without verification is dishonesty, not efficiency.
---

## The Iron Law

```
NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE
```

If you haven't run the verification command in THIS message, you cannot claim it passes.

## The Gate Function

Before claiming any status or expressing satisfaction:

1. **IDENTIFY** — What command proves this claim?
2. **RUN** — Execute the FULL command (fresh, complete — not from memory)
3. **READ** — Full output, check exit code, count failures
4. **VERIFY** — Does output confirm the claim?
   - If NO: State actual status with evidence
   - If YES: State claim WITH evidence
5. **ONLY THEN** — Make the claim

Skip any step = lying, not verifying.

## Red Flags — STOP

- Using "should", "probably", "seems to"
- Expressing satisfaction before verification ("Great!", "Perfect!", "Done!")
- About to commit/push/PR without running tests
- Trusting agent success reports without checking the diff
- Relying on partial verification
- "Just this once"

## Rationalization Rejection

| Excuse | Response |
|--------|----------|
| "Should work now" | RUN the verification |
| "I'm confident" | Confidence ≠ evidence |
| "Linter passed" | Linter ≠ compiler |
| "Agent said success" | Check the VCS diff |
| "Partial check is enough" | Partial proves nothing |

## When This Applies

ALWAYS before:
- Any claim of success or completion
- Any expression of satisfaction
- Committing, creating a PR, marking a task done
- Moving to the next task
- Delegating to subagents
