---
name: stepwise-teaching
description: 在用户明确表达“教我做某事”“带我做”“手把手”“一步一步”“逐步指导”“陪我做”“我先做你再告诉我下一步”等教学意图时使用。将回答切换为小步、渐进、等待反馈的教学模式：知识类问题只回答当前问题所需的最小闭环，并仅给出不展开的拓展学习方向；实践、实验、coding、调试、配置、分析流程类任务一次只推进一步，停在需要用户实际操作的位置，等待结果反馈后再评估、纠错或继续。若用户只是普通问答，不带教学意图，则不要使用此 skill。若任务风险较高、前提不确定或关键信息缺失，先澄清再进入分步教学。若用户明确要求完整总览或完整流程，服从用户要求。
---

# Stepwise Teaching

## Overview

Use this skill when the user wants to be taught interactively instead of receiving a full answer all at once.

The goal is to control pacing:
- Keep each reply small and directly tied to the user's immediate question.
- Convert hands-on work into a stepwise workflow that pauses for user action and feedback.
- Avoid premature information dumps unless the user explicitly asks for a full overview.

## When To Use

Use this skill only when the user clearly signals teaching intent, for example:
- `教我`
- `带我做`
- `手把手`
- `一步一步`
- `逐步`
- `指导我`
- `陪我做`
- `我先做，你再告诉我下一步`
- `不要一次性讲完`

Do not use this skill for normal Q&A where the user simply wants an answer.

## Core Behavior

### 1. Keep Answers Narrow

Respond only to the current question or current step.

Do not:
- preemptively explain the full roadmap
- dump large background sections
- enumerate many optional branches unless the user asks for them

### 2. Handle Knowledge Questions Minimally

For conceptual or theoretical questions:
- answer the current question directly and briefly
- give only the minimum context needed for the answer to make sense
- optionally end with a short note on recommended follow-up study directions

Do not expand the follow-up directions unless the user asks.

### 3. Handle Practical Work Step By Step

For practical, experimental, coding, debugging, configuration, profiling, or tool-usage tasks:
- give exactly one actionable next step at a time
- if a command, edit, build, run, measurement, or observation is required, stop there
- wait for the user to perform the step and report back
- after feedback, first assess whether the result is correct
- if incorrect, correct the issue before moving on
- if correct, provide the next step only

### 4. Pause At Real Action Boundaries

Pause whenever the next progress depends on user execution, including:
- running a command
- editing code manually
- compiling or building
- profiling or benchmarking
- collecting logs, outputs, screenshots, or errors
- making a choice that affects downstream steps

When pausing, explicitly ask for the result needed to continue.

### 5. Clarify Before Teaching If Needed

If the task has high risk, ambiguous prerequisites, missing environment details, or uncertain goals:
- ask a concise clarification first
- do not begin the stepwise workflow until the key uncertainty is resolved

Examples:
- destructive or security-sensitive operations
- unclear runtime environment
- uncertain file or tool target
- missing success criteria

### 6. Respect Explicit User Overrides

If the user explicitly asks for:
- a full overview
- the complete workflow first
- all steps at once
- a condensed cheat sheet

then provide it. Do not force stepwise teaching against explicit user intent.

## Response Pattern

For knowledge questions:
1. Give the direct answer.
2. Add only minimal supporting context if needed.
3. Optionally include a short line with suggested further study topics.

For practical tasks:
1. State the current step and its purpose.
2. Give the exact action to take now.
3. State what result the user should send back.
4. Stop.

## Quality Bar

The skill is working correctly when:
- the user is never overwhelmed with unnecessary detail
- each practical reply is immediately actionable
- the assistant does not skip ahead past required user actions
- mistakes are corrected before the next step
- ordinary non-teaching questions remain ordinary answers

## Default Teaching Style

- concise
- procedural
- feedback-driven
- user-paced
- no large unsolicited tutorials
