---
name: test-driven-development
description: Enforce RED-GREEN-REFACTOR TDD cycles. Use when implementing any new feature or fixing any bug. If you didn't watch the test fail, you don't know if it tests the right thing.
---

## Core Principle

**"If you didn't watch the test fail, you don't know if it tests the right thing."**

This applies universally: new features, bug fixes, refactoring, and behavior modifications.

## The Cycle

**RED** — Write a test demonstrating desired functionality before any implementation exists. Run it. Confirm it fails for the expected reason.

**GREEN** — Write the simplest possible code to make the test pass. Run tests. Confirm they pass.

**REFACTOR** — Improve code quality while keeping tests green. Run tests again to confirm.

## Non-Negotiable Requirements

- After writing tests: confirm they FAIL for expected reasons
- After implementing: verify ALL tests pass
- During refactor: ensure tests remain green

## Common Rationalizations to Reject

- "I already manually tested it" → RUN the automated test
- "I'll write tests afterward" → No. Tests first.
- "It's a simple change" → Still requires RED-GREEN
- "Tests written after pass immediately" → That proves nothing

## The Fundamental Rule

**Production code → test exists and failed first. Otherwise → not TDD.**
