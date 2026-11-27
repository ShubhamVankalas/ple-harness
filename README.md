# ple-engineer

An agent harness for [Physical-Engineering's Last Exam](https://github.com/google-research/ple).
It wraps the reference tool loop with the working habits of an engineering
office: know every file in the project folder, read drawings and models with
proper tools, measure what you are about to issue, and have someone else check
it against the source documents before it goes out.

Author: Shubham Vankalas. License: Apache-2.0.

## Why these pieces

Each component answers a failure we measured on the public dev set with the
plain reference loop:

| Failure seen on dev runs | Component |
|---|---|
| Deliverable crashes or fails the grader's reader (untyped IFC values, wrong argument counts, units not attached, schema rule broken) | **Format gate**: strict readers, schema validation, run delivered scripts the way the task runs them |
| Model "self-checks" share the bug of the script that wrote the file | **Measured facts**: the harness measures the finished file itself (unit scale, real extents, typing, overlaps) and diffs it against any example the task ships |
| Decisive input never opened (a calendar, an exhibit, a later letter) | **Survey with previews** and a **coverage ledger** of unopened files |
| A reviewer that only sees the output finds nothing real | **Independent checker**: a fresh-context agent with read-only tools that re-derives results from the inputs |
| Runs stop after minutes of a six-hour budget, or burn hundreds of steps patching one file | Staged finish (gates before the answer is accepted) and a **churn governor** |
| Task container lacks parsers; drawings have no text layer | Harness-side **inspect** and **view** tools |

## Architecture

```
            task instruction
                   |
        +----------v-----------+
        | survey               |  every file: size, type, first text
        +----------+-----------+
                   |
        +----------v-----------+      tools in the task container:
        | main agent loop      |<---> bash, read_file, write_file, edit_file, MCP
        | (serve.py loop +     |<---> harness tools: inspect, view
        |  engineering prompt) |      churn governor on every result
        +----------+-----------+
                   | answer
        +----------v-----------+  fail
        | 1 deliverable gate   |--------> back to the agent
        | 2 format gate        |  strict readers, run delivered scripts,
        |   + measured facts   |  measurements, example diff
        | 3 coverage ledger    |  unopened inputs, with previews
        | 4 independent check  |  fresh agent, read-only tools, evidence
        +----------+-----------+
                   | clean, or out of budget
               final answer
```

## Files

| File | Purpose |
|---|---|
| `harness.py` | `EngineerAgent`: survey, harness tools, gates, coverage ledger, independent checker, churn governor |
| `localtools.py` | Parsing, rendering, validation and measurement of PDF, Office, DXF and IFC files inside the harness container |
| `serve.py` | Reference agent loop: container and MCP tools, context compaction, deadline handling (Google LLC, Apache-2.0) |
| `taskbridge.py` | Serve protocol (Google LLC, Apache-2.0) |
| `submission.yaml` | Model, build and start commands |
| `requirements.txt` | Pinned dependencies |

## Components

**Survey.** Before the first model call the harness lists every project file
with its size and the first words of its content, and reports which document
libraries the task container has. Titles, revision marks and dates in the
previews let the model pick governing documents and spot superseded ones
without opening everything first.

**inspect / view.** `inspect` extracts PDF text and tables per page, XLSX cells
with formulas and values, DOCX and PPTX text, DXF layers, texts and dimensions,
and IFC structure. `view` renders a PDF page, image or DXF as a picture, with a
crop that zooms and upscales small regions. Both run in the harness container
on a copy of the file, so they work when the task container has no parsers.

**Deliverable gate.** Output paths named in the instruction (absolute or
relative to the working directory) must exist before the answer is accepted.

**Format gate and measured facts.** Every file written during the task is
opened with a strict reader (IFC with schema and EXPRESS rules, JSON, CSV,
XLSX, DOCX, PPTX, PDF, DXF, ZIP). A delivered generator script is run the way
the task says it will be run and its output is validated too. For structured
models the harness reports what a consumer would measure: the unit scale and
whether it is attached to the project, real extents in metres per class,
typing, untyped property values, and coarse overlaps. If the task ships an
example of the same format, structural differences from it are listed.

**Coverage ledger.** Tool calls are tracked per input file. Before finishing,
the agent gets the list of files it never touched, each with its preview.

**Independent checker.** A separate agent with a fresh context and read-only
tools receives the task, the summary, the measurements and the unopened-file
list. It opens the deliverables as their consumer would, runs the validations
the task prescribes, re-derives key numbers from the governing inputs and
reports defects with evidence. The main agent verifies each against the inputs
and fixes the valid ones. Up to `REVIEW_ROUNDS` rounds, time permitting.

**Churn governor.** Repeated identical errors, many rewrites of one file and
long call counts trigger short, specific advice appended to the tool result.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `REVIEW_ROUNDS` | 2 | independent-check rounds |
| `AUDIT_STEPS` | 45 | tool calls per check |
| `AUDIT_SHARE` | 0.3 | share of remaining time one check may use (max 45 min) |
| `REVIEW_MIN_LEFT_MIN` | 20 | no check starts with less time left |
| `KEEP_IMAGES` | 6 | images kept in context |
| `COMPACT_AT_TOKENS` | 300000 | context size that triggers compaction |
| `PREVIEW_FILES` | 90 | files previewed by the survey |

## Run

```
uv run python -m kit check /path/to/ple-harness
uv run python eval/run.py /path/to/ple-harness --env-file local.env
```

The harness uses only the task container, the task's MCP servers and the model
endpoint. It never reads grader or test files and stores nothing about tasks.
