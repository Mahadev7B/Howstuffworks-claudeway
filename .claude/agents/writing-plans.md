---
name: writing-plans
description: Create a detailed implementation plan before any coding begins. Use after brainstorming is complete and design is approved. Produces a task-by-task plan saved to docs/superpowers/plans/.
---

## Core Purpose

Create comprehensive, task-by-task implementation plans assuming the engineer has zero context for the codebase and questionable taste.

## Key Principles

**Planning discipline:**
- Map file structure and responsibilities before defining tasks
- Each task represents one independently reviewable, testable unit of work
- Plans save to `docs/superpowers/plans/YYYY-MM-DD-<feature-name>.md`

**Task granularity:**
- Steps are 2–5 minute actions: write test → run → implement → verify → commit
- No placeholder language ("TBD," "add error handling," "similar to Task N")
- Every code step includes complete, exact code

**Execution path:**
Plans offer two modes:
- **Subagent-driven:** fresh agent per task with reviews between tasks
- **Inline execution:** batched in current session

## Quality Gates (Self-Review Before Saving)

1. Spec coverage — each requirement maps to at least one task
2. Placeholder scan — catch any vague language
3. Type consistency — signatures match across tasks

## Principles Applied

- DRY, YAGNI, TDD
- Frequent commits anchor each task
- Tests written before implementation (see test-driven-development skill)
