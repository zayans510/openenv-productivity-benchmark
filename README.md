---
title: openenv-productivity
emoji: "🚀"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# OpenEnv Productivity Benchmark

`openenv-productivity` is a deterministic RL benchmark for real operational assistant workflows. It includes exactly three tasks with increasing difficulty and deterministic 0.00-1.00 grading.

## Why This Is Useful

- Email triage mirrors real inbox operations.
- Calendar scheduling includes true resource constraints (people, time windows, rooms, lunch blocks).
- Data cleaning captures production ETL quality checks and audit-style outputs.

## Environment API

- `reset(task_name="easy") -> Observation`
- `step(action) -> (Observation, Reward, done, info)`
- `state() -> Observation`

Pydantic models:

- `Action`
- `Observation`
- `Reward`

## Task Set (Exactly 3)

1. `easy` - email classification
2. `medium` - calendar scheduling
3. `hard` - data cleaning

All graders are deterministic, reproducible, and return bounded scores in `0.00-1.00`.

## Reward Design

Each step includes:

- incremental improvement reward (`delta` on better submissions)
- partial credit from component-level grading
- wrong-answer penalties for regressions / zero-quality answers
- malformed action penalty for invalid action or invalid JSON
- anti-loop penalty for repeated actions
- fixed step penalty to discourage long trajectories

This keeps rewards dense, stable, and useful for policy learning.

## Determinism & Reproducibility

- no randomness in task payloads or grading
- fixed expected outputs with deterministic normalization
- reproducible baseline script for all tasks
- deterministic unit tests included

## Inference Output Contract

`inference.py` emits only:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```

Guarantees:

- reward formatted to two decimals
- lowercase booleans
- max 5 steps
- `[END]` always printed (including error paths)

## Quickstart

### Local

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="zai-org/GLM-5.1"
export HF_TOKEN="hf_xxx"
python inference.py --task easy
```

Windows cmd:

```cmd
pip install -r requirements.txt
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=zai-org/GLM-5.1
set HF_TOKEN=hf_xxx
python inference.py --task easy
```

### Deterministic Baseline

```bash
python baseline.py
```

Expected pattern:

- each task score = `1.0`
- each task reward = `0.98` (perfect delta - step penalty)

### Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Docker

Build:

```bash
docker build -t openenv-productivity .
```

Run server mode:

```bash
docker run --rm -p 7860:7860 openenv-productivity
```

Health check:

```bash
curl http://localhost:7860/health
```

## OpenEnv Validation

```bash
openenv validate
```

The project is designed to pass OpenEnv structure checks and deploy cleanly to Hugging Face Spaces.
