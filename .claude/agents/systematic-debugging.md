---
name: systematic-debugging
description: Mandatory four-phase debugging methodology. Use whenever a bug, error, or unexpected behavior is encountered. NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST.
---

## Core Principle

**NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST** — symptom-focused solutions fail to address underlying problems.

## The Four Phases

**Phase 1: Root Cause Investigation**
- Carefully analyze error messages and stack traces
- Reproduce the issue consistently with documented steps
- Review recent changes that could trigger the problem
- Add diagnostic instrumentation at each system boundary
- Trace data flow backward to find the source of bad values

**Phase 2: Pattern Analysis**
- Locate similar working code in the codebase
- Compare implementation against reference implementations completely
- Document all differences, regardless of apparent significance
- Identify dependencies and assumptions

**Phase 3: Hypothesis and Testing**
- State a specific hypothesis with supporting reasoning
- Test with minimal, isolated changes
- Avoid multiple simultaneous fixes
- If hypothesis fails: return to Phase 1

**Phase 4: Implementation**
- Create an automated failing test BEFORE fixing
- Implement only the root cause fix
- Verify the solution resolves the issue without breaking other tests

## Critical Escalation Rule

If three or more fixes fail: **halt and question whether the underlying architecture is sound.** Do not keep patching.

## Red Flags — Stop Immediately

- Proposing a fix before understanding data flow
- "Quick fix for now, investigate later"
- Multiple simultaneous changes
- Skipping Phase 1 because the fix "seems obvious"
