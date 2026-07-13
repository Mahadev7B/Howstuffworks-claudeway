---
name: brainstorming
description: Structured ideation skill. Use when starting any new feature, app, or design. Activates before writing code — refines rough ideas through questions, explores alternatives, presents design in sections for validation. Hard gate: do NOT write any code until design is approved.
---

## Core Principle

**Hard gate:** Do NOT invoke any implementation skill, write any code, scaffold any project, or take any implementation action until you have presented a design and the user has approved it.

## Process (Nine Sequential Steps)

1. Explore project context — check existing files and documentation
2. Offer visual tools only when genuinely helpful for a specific question
3. Ask clarifying questions **one at a time** to understand purpose and constraints
4. Propose 2–3 alternative approaches with trade-offs
5. Present design sections and gather approval after each
6. Write the design document to `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`
7. Self-review the spec for placeholders, contradictions, and ambiguity
8. Have the user review the written specification
9. Invoke the writing-plans skill to create an implementation plan

## Design Principles

- Smaller, well-bounded units are easier to reason about — each component should have one clear purpose with well-defined interfaces
- Prefer multiple-choice questions over open-ended ones
- One question per message
- No upfront visual companion offering — only when visually appropriate

## After Brainstorming

The only skill invoked after brainstorming concludes is **writing-plans**. Never jump directly to implementation.
