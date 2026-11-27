"""Engineering agent harness for Physical-Engineering's Last Exam.

Builds on the reference tool loop in serve.py (Apache-2.0, Google LLC) and adds:

1. A workspace survey before the first model call: every project file with its
   size and type, and which document libraries the task container has.
2. Harness-side document tools. `inspect` parses PDF, Office, DXF and IFC files
   in this container (pinned libraries), so the model reads drawings,
   schedules and models even when the task container has no parsers. `view`
   renders a PDF page, image or DXF (optionally a zoomed crop) and shows it to
   the model as an image, for drawing callouts that have no text layer.
3. A deliverable gate: required output paths are parsed from the instruction
   and checked before the agent may finish.
4. A format gate: delivered files are opened with strict readers, delivered
   generator scripts are run as the grader would run them, and structured
   outputs are measured (units, extents, types) and compared with any example
   file the task ships.
5. A coverage ledger: input files no tool call touched are listed back to the
   agent, with a preview, before it may finish.
6. An independent check: a fresh-context agent with read-only tools re-derives
   the key results from the inputs, runs the prescribed validations and lists
   defects with evidence; the main agent then fixes them. This repeats while
   time and the review budget allow.
"""

from __future__ import annotations

import asyncio
import json
import os
import posixpath
import re
import shlex
import sys
from typing import Any

import openai
import serve
from serve import Agent, CutOff, OutOfTime, parse_arguments, tool, visible
import taskbridge
import localtools

REVIEW_ROUNDS = int(os.environ.get("REVIEW_ROUNDS", "2"))
# a review starts only if this many minutes remain
REVIEW_MIN_LEFT_MIN = float(os.environ.get("REVIEW_MIN_LEFT_MIN", "20"))
KEEP_IMAGES = int(os.environ.get("KEEP_IMAGES", "6"))
IMAGE_TOKENS = 1600  # rough prompt cost of one image
REVIEW_CHARS = int(os.environ.get("REVIEW_CHARS", "120000"))
AUDIT_STEPS = int(os.environ.get("AUDIT_STEPS", "45"))  # tool calls one audit may make
AUDIT_SHARE = float(os.environ.get("AUDIT_SHARE", "0.3"))  # of the time left, per audit
PREVIEW_FILES = int(os.environ.get("PREVIEW_FILES", "90"))
DOC_EXT = (".pdf", ".docx", ".pptx", ".xlsx", ".xlsm", ".xls", ".dxf", ".ifc", ".zip")
TEXT_EXT = (".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml", ".eml", ".log", ".ini", ".toml", ".py", ".html",
            ".htm", ".tsv", ".inp", ".dat", ".cfg", ".tex", ".rst")
SURVEY_FILES = 400

EXTRA_SYSTEM = """

Engineering practice (you are graded like a licensed engineer's deliverable):
- First make a checklist of every deliverable, required file path, format, section, and number the task asks for, \
and of every input file that bears on it. Satisfy each item explicitly.
- Project folders contain distractors (superseded revisions, unrelated sheets, old e-mails). Decide which document and \
revision governs (latest issue/approval, the one the task names) before using any value from it.
- Read the governing documents (specifications, code excerpts, criteria, approval letters, schedules) before you \
compute. Use the project's own criteria, values and clause numbers over general rules of thumb.
- Keep a requirements register: each requirement you apply, with its source file, clause/section and value. When a \
deliverable has room for it (or the task asks for the requirements applied), include that register in it.
- Design tasks are graded on constraints and an objective: every hard constraint (safety, code limits, physical \
feasibility, interfaces between documents) must hold, or the design earns no credit for its objective. So: write a \
checker script that tests your design against every constraint you found, get it fully passing, and only then improve \
the objective (e.g. lower cost or length) while re-running the checker after every change.
- Drawings: use the inspect tool for the text layer, and the view tool to look at sheets, zooming (crop) into title \
blocks, callouts, schedules and dimensions. Never guess a value that a drawing shows.
- Compute with code (python3 in the container), keep units explicit and convert carefully; sanity-check magnitudes \
against typical values and against the inputs.
- In reports and memos, state the key numbers (with units) and the decision or verdict plainly, with the governing \
check; graders look for load-bearing numbers and conclusions. Follow any requested addressee, date, format, and file \
name exactly.
- Write structured deliverables (IFC, XLSX, DOCX, DXF, PDF) with the format's own library, never as hand-typed \
text: e.g. ifcopenshell.api (root.create_entity, aggregate, spatial, geometry; real GUIDs; every product placed; \
coordinates in the project's units and at the real scale of the inputs). Re-open each file with a strict reader and \
check its contents before you finish; a file that fails to open scores zero. Run every validation the task itself \
prescribes, with exactly the settings it names; if the deliverable is a script, run it end to end the way the task \
says it will be run and validate what it writes.
- Keep scratch work (helper scripts, notes, test outputs) under /tmp/work, not in the project or output folders.
- Keep a running notes file /tmp/work/NOTES.md: values read from each drawing or document (with sheet/page), unit \
conventions, decisions, validation commands and their latest result. Update it as you go and re-read it after your \
context is compacted, instead of re-deriving or re-reading everything.
- A self-check only counts if it measures the finished file the way its consumer will (re-open it; measure real \
extents and units; run the prescribed validator), not the numbers you intended to write. Never report a check as \
passed when its command errored or printed a failure: fix it or say plainly that it fails.
- Before finishing, skim every input file you have not opened: decisive facts (a calendar, an exhibit, a later \
letter, a checklist) are often in files whose names look irrelevant.
- When evidence a governing document requires does not exist in the project, say so explicitly instead of \
substituting something else; say who must decide what, and by which real date.
- Never stop early: you have hours. Use the time to verify, cross-check against the inputs, and polish.

Harness tools (they run outside the container, on a copy of the file):
- inspect: text of a PDF (per page, with tables; or search with query), DOCX, PPTX, XLSX (formulas and values), DXF \
(layers, texts, dimensions), IFC (entities), ZIP listings.
- view: look at a PDF page, an image or a DXF drawing; crop=[x0,y0,x1,y1] (fractions of the page, origin top-left) \
zooms in.
In inspect output, lines in [[double brackets]], [style] prefixes and " | " separators are annotations, not file text."""

HARNESS_TOOLS = [
    tool(
        "inspect",
        "Extract the content of a document in the task container: PDF text"
        " and tables per page (pages='1-3,7'), or the pages' lines matching"
        " query; XLSX cells with formulas and values (sheet, cells='A1:H40');"
        " DOCX/PPTX text and tables; DXF layers, texts and dimensions; IFC"
        " entities; ZIP listings. Use it instead of writing parsers.",
        ["path"],
        path={"type": "string"},
        pages={"type": "string", "description": "PDF pages, e.g. '1-3,7'"},
        sheet={"type": "string", "description": "XLSX sheet name"},
        cells={"type": "string", "description": "XLSX range, e.g. 'A1:K60'"},
        query={"type": "string", "description": "only lines containing this"},
    ),
    tool(
        "view",
        "Look at a PDF page, an image file, or a DXF drawing as a picture. Use"
        " crop to zoom into part of it: [x0, y0, x1, y1] as fractions of the"
        " width and height, origin at the top-left (e.g. [0.6, 0.7, 1, 1] is"
        " the bottom-right corner where title blocks usually sit).",
        ["path"],
        path={"type": "string"},
        page={"type": "integer", "description": "1-based PDF page"},
        crop={"type": "array", "items": {"type": "number"}},
        dpi={"type": "integer", "description": "render resolution (default 110)"},
    ),
]

AUDIT_SYSTEM = """You are a senior licensed professional engineer performing the independent check of a colleague's \
deliverables before they are issued. You work in the same Linux container, with read-only intent: tools bash (starts in \
{workdir}), read_file, inspect and view. Do NOT modify or delete any project input or deliverable; write scratch \
files only under /tmp/audit. You have about {minutes} minutes and {steps} tool calls.

You are given the task as the client wrote it, the colleague's summary, the harness's measurements of the delivered \
files, and the list of input files the colleague never opened. Check the work against the INPUT FILES and the task, \
not against the colleague's own claims:
1. Open each deliverable the way its consumer will (the stated reader, validator or run command, with exactly the \
settings the task names) and measure it: does it open, are units/scale/extents/counts physically right, does every \
prescribed check pass? If the task prescribes a validation, run it yourself.
2. Re-derive the load-bearing numbers, verdicts and dates independently from the governing input documents (open \
them; decide which revision governs). Report any number, unit, sign, direction (conservative vs non-conservative), \
date or conclusion that the inputs contradict.
3. Skim the unopened input files; report any that change the answer (quote the passage).
4. Walk the task text requirement by requirement; report anything not explicitly delivered, misnamed, misplaced, \
or hedged where a decision is required.
5. For designs: test every hard constraint on the finished file with your own script; a constraint failure matters \
more than any objective.

Then answer without calling a tool. List only concrete defects that would cost credit with a strict grader, most \
important first; for each give the file and location, the evidence (a quoted passage, or the command you ran and \
its output), and the exact fix. Do not report cosmetic points, and do not report anything you could not verify. In \
tool output, lines in [[double brackets]], [style] prefixes and " | " separators are extraction annotations, not \
file text. If you verified the work and found no defects, answer exactly NO_ISSUES."""

REL_RE = re.compile(r"`((?:\.?/)?[\w@+-][\w./@+-]*\.[A-Za-z0-9]{1,5})`")
serve.SUMMARY_PROMPT += (
    " Include: every value you read from a drawing or document with its sheet/page; unit and sign conventions;"
    " which document revisions govern; the exact validation commands and their latest results; input files not yet"
    " opened; and the path of your notes file.")

PATH_RE = re.compile(r"(?<![\w.])(/(?:app|filesystem|workspace|output|data|home|mnt|tmp)/[\w./@+-]*[\w/])")


def wanted_paths(instruction: str, workdir: str = "") -> list[str]:
  """Paths the instruction names (likely deliverables or inputs); relative ones resolved to workdir."""
  seen: list[str] = []
  for m in PATH_RE.finditer(instruction):
    p = m.group(1).rstrip(".,;:)")
    if p not in seen:
      seen.append(p)
  if workdir:
    for m in REL_RE.finditer(instruction):
      p = m.group(1)
      if p.startswith("/") or "/" not in p and not re.search(
          r"(save|write|create|produce|deliver|submit|output)[^.\n]{0,80}`" + re.escape(p), instruction, re.I):
        continue
      p = posixpath.normpath(posixpath.join(workdir, p))
      if p not in seen:
        seen.append(p)
  return seen[:60]


class EngineerAgent(Agent):
  """Reference agent plus survey, document tools, deliverable gate and review."""

  def __init__(self, session: taskbridge.Session):
    super().__init__(session)
    self.tools += HARNESS_TOOLS
    self.pending_images: list[tuple[str, bytes]] = []
    self.image_msgs: list[dict[str, Any]] = []
    self.reviews = 0
    self.gate_checks = 0
    self.format_checks = 0
    self.cache: dict[str, bytes] = {}
    self.inputs: list[tuple[int, str]] = []  # (bytes, path) seen by the survey
    self.opened: set[str] = set()
    self.coverage_checks = 0
    self.read_only = False

  # ------------------------------------------------------------ setup

  async def survey(self) -> str:
    """A listing of the project files and the container's tooling."""
    wd = self.session.workdir or "/app"
    roots = " ".join(shlex.quote(r) for r in dict.fromkeys([wd, "/filesystem", "/app"]))
    cmd = (
        f"for r in {roots}; do [ -d \"$r\" ] && find \"$r\" -xdev -type f"
        " -not -path '*/.git/*' -not -path '*/node_modules/*' -not -path"
        " '*/__pycache__/*' -printf '%s\\t%p\\n' 2>/dev/null; done | sort -u -k2"
        f" | head -n {SURVEY_FILES}; echo '---';"
        " python3 - <<'PY' 2>/dev/null || echo 'python3: not available'\n"
        "import importlib.util as u, sys\n"
        "mods=['numpy','scipy','pandas','openpyxl','docx','pptx','reportlab',"
        "'fpdf','pdfplumber','pypdf','fitz','matplotlib','ezdxf','ifcopenshell',"
        "'shapely','sympy','xlsxwriter','PIL']\n"
        "print('python', sys.version.split()[0], '| modules:', ' '.join("
        "m for m in mods if u.find_spec(m)), '| missing:', ' '.join("
        "m for m in mods if not u.find_spec(m)))\n"
        "PY\n"
        "for t in libreoffice soffice pandoc pdftotext qpdf gs node; do command -v $t"
        " >/dev/null && printf '%s ' $t; done; echo"
    )
    try:
      res = await self.session.exec(cmd, 60)
      out = res.get("output") or ""
    except taskbridge.TaskError as exc:
      return f"(survey failed: {exc})"
    files, _, tooling = out.partition("---")
    lines = []
    for row in files.strip().splitlines():
      size, _, path = row.partition("\t")
      try:
        kb = int(size) / 1024
      except ValueError:
        continue
      lines.append(f"{kb:9.1f} KB  {path}")
      self.inputs.append((int(size), path))
    named = wanted_paths(self.session.instruction, self.session.workdir or '')
    named_note = ""
    if named:
      named_note = "\nPaths named in the task: " + ", ".join(named)
    try:
      previews = await asyncio.wait_for(self.previews(), 240)
    except Exception as exc:  # pylint: disable=broad-except
      await self.session.log(f"survey: previews failed: {type(exc).__name__}: {str(exc)[:200]}")
      previews = {}
    self.preview_map = previews
    if previews:
      lines = [ln + ("   :: " + previews[ln.split(" KB  ", 1)[1]] if ln.split(" KB  ", 1)[1] in previews else "")
               for ln in lines]
    return (
        f"Project files ({len(lines)} shown; ':: ' is the first text of the file, for triage only):\n"
        + "\n".join(lines)
        + f"\n\nTask container tooling: {tooling.strip()}"
        + "\n(Office/PDF libraries missing in the container? Use inspect/view to read files, and write"
          " deliverables with whatever is available, e.g. python-docx or openpyxl if present, otherwise build"
          " the Office XML zip with python's zipfile.)"
        + named_note
    )

  async def previews(self) -> dict[str, str]:
    """First text of each input document (path -> preview), bounded in count, size and time."""
    import time  # pylint: disable=import-outside-toplevel
    out: dict[str, str] = {}
    texts = [p for sz, p in self.inputs if localtools.ext(p) in TEXT_EXT][:PREVIEW_FILES]
    if texts:
      cmd = "; ".join(
          f"printf '\\n@@%s@@ ' {i}; head -c 400 {shlex.quote(p)} | tr '\\n\\r\\t' '   ' | cut -c1-230"
          for i, p in enumerate(texts))
      res = await self.session.exec(cmd, 60)
      for m in re.finditer(r"@@(\d+)@@ ([^\n]*)", res.get("output") or ""):
        text = " ".join(m.group(2).split())
        if text:
          out[texts[int(m.group(1))]] = text[:220]
    t0, budget = time.time(), 250_000_000
    docs = [(sz, p) for sz, p in self.inputs
            if localtools.ext(p) in DOC_EXT or localtools.ext(p) in localtools.IMAGE_EXT]
    for sz, p in docs[:PREVIEW_FILES]:
      if time.time() - t0 > 180 or sz > 30_000_000 or sz > budget:
        continue
      budget -= sz
      try:
        data = await self.fetch(p)
        out[p] = await asyncio.wait_for(asyncio.to_thread(localtools.preview, p, data), 30)
      except Exception:  # pylint: disable=broad-except
        continue
    return out

  def unopened(self) -> list[tuple[int, str]]:
    """Input files no tool call has touched."""
    return [(sz, p) for sz, p in self.inputs if p not in self.opened and sz > 0]

  def note_opened(self, name: str, args: dict[str, Any]) -> None:
    if name in ("read_file", "inspect", "view"):
      self.opened.add(self.path(str(args.get("path", ""))))
    elif name == "bash":
      cmd = str(args.get("command", ""))
      for _, p in self.inputs:
        if p in self.opened:
          continue
        base = posixpath.basename(p)
        if p in cmd or (len(base) >= 6 and base in cmd):
          self.opened.add(p)

  async def read_file(self, path: str) -> str:
    out = await super().read_file(path)
    e = localtools.ext(path)
    if " is a binary file (" in out[:400] and (e in DOC_EXT or e in localtools.IMAGE_EXT):
      return (f"{path} is a binary {e} file; showing the inspect tool's extraction instead"
              " (use inspect for pages/sheets/queries, view to look at it):\n" + await self.tool_inspect(path))
    return out

  # ------------------------------------------------------------ files

  async def fetch(self, path: str) -> bytes:
    path = self.path(path)
    if path in self.cache:
      return self.cache[path]
    result = await self.session.read_result(path)
    if not result.get("ok"):
      raise taskbridge.TaskError(result.get("error") or f"cannot read {path}")
    if not result.get("truncated"):
      import base64  # pylint: disable=import-outside-toplevel

      data = base64.b64decode(result["content_b64"])
    else:
      data = await self.fetch_parts(path)
    if len(self.cache) > 40:
      self.cache.pop(next(iter(self.cache)))
    self.cache[path] = data
    return data

  async def fetch_parts(self, path: str) -> bytes:
    """Reads a file over 10 MB by splitting it in the container."""
    tmp = "/tmp/.harness_parts"
    q = shlex.quote(path)
    res = await self.session.exec(
        f"rm -rf {tmp} && mkdir -p {tmp} && split -b 8M -d -a 3 {q} {tmp}/p_ && ls {tmp}", 120)
    parts = sorted((res.get("output") or "").split())
    data = b""
    for p in parts:
      data += await self.session.read_file(f"{tmp}/{p}")
    await self.session.exec(f"rm -rf {tmp}", 30)
    return data

  def forget(self, path: str) -> None:
    self.cache.pop(self.path(path), None)

  # ------------------------------------------------------------ tools

  async def run_tool(self, name: str, args: dict[str, Any]) -> str:
    if self.read_only and name in ("write_file", "edit_file"):
      return "error: the checker must not change files; report the defect and its fix instead"
    if isinstance(args, dict):
      self.note_opened(name, args)
    if name in ("write_file", "edit_file", "bash"):
      # a write may change any cached file
      self.cache.clear()
    if name == "inspect":
      return await self.tool_inspect(**args)
    if name == "view":
      return await self.tool_view(**args)
    return await super().run_tool(name, args)

  async def tool_inspect(self, path: str = "", pages: Any = None, sheet: Any = None,
                         cells: Any = None, query: Any = None, **_: Any) -> str:
    await self.session.log(f"inspect {path} pages={pages} sheet={sheet} query={query}")
    try:
      data = await self.fetch(path)
      return await asyncio.to_thread(localtools.inspect, path, data, pages, sheet, cells, query)
    except taskbridge.TaskError as exc:
      return f"error: {exc}"
    except Exception as exc:  # pylint: disable=broad-except
      return f"error: could not inspect {path}: {type(exc).__name__}: {exc}"

  async def tool_view(self, path: str = "", page: Any = None, crop: Any = None,
                      dpi: Any = None, **_: Any) -> str:
    await self.session.log(f"view {path} page={page} crop={crop}")
    try:
      data = await self.fetch(path)
      img = await asyncio.to_thread(localtools.render, path, data, page, crop, dpi)
    except taskbridge.TaskError as exc:
      return f"error: {exc}"
    except Exception as exc:  # pylint: disable=broad-except
      return f"error: could not render {path}: {type(exc).__name__}: {exc}"
    label = f"{path}" + (f" page {page}" if page else "") + (f" crop {crop}" if crop else "")
    self.pending_images.append((label, img))
    return f"rendered {label}; the image follows this result."

  def attach_images(self) -> None:
    if not self.pending_images:
      return
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Images you asked to view: "
         + "; ".join(lbl for lbl, _ in self.pending_images)}
    ]
    for _, img in self.pending_images:
      content.append({"type": "image_url", "image_url": {"url": localtools.data_url(img)}})
    msg = {"role": "user", "content": content}
    self.messages.append(msg)
    self.image_msgs.append(msg)
    self.pending_images = []
    self.prune_images()

  def prune_images(self) -> None:
    live = {id(m) for m in self.messages}
    self.image_msgs = [m for m in self.image_msgs if id(m) in live and isinstance(m["content"], list)]
    count = 0
    for m in reversed(list(self.image_msgs)):
      imgs = [c for c in m["content"] if c.get("type") == "image_url"]
      if count + len(imgs) > KEEP_IMAGES:
        text = m["content"][0]["text"]
        m["content"] = text + " [image removed to save context; view it again if needed]"
        self.image_msgs.remove(m)
      else:
        count += len(imgs)

  def estimate_prompt_tokens(self) -> int:
    total = 0
    for m in self.messages:
      if isinstance(m.get("content"), list):
        for c in m["content"]:
          if c.get("type") == "image_url":
            total += IMAGE_TOKENS * 4
          else:
            total += len(json.dumps(c))
        total += 40
      else:
        total += len(json.dumps({k: v for k, v in m.items() if k not in ("output_items",)}))
    return total // 4

  # ------------------------------------------------------------ gate & review

  async def missing_outputs(self) -> list[str]:
    """Paths the task names that look like outputs and do not exist."""
    named = wanted_paths(self.session.instruction, self.session.workdir or '')
    if not named:
      return []
    quoted = " ".join(shlex.quote(p) for p in named)
    res = await self.session.exec(
        f"for p in {quoted}; do [ -e \"$p\" ] || echo \"$p\"; done", 30)
    gone = (res.get("output") or "").split()
    # only paths that look like outputs: under an output dir or named deliverable
    instr = self.session.instruction.lower()
    out = []
    for p in gone:
      low = p.lower()
      if "output" in low or "deliver" in low or "submission" in low or re.search(
          r"(save|write|create|produce|deliver|submit|place|put)[^.\n]{0,120}" + re.escape(low), instr):
        out.append(p)
    return out

  async def new_files(self) -> list[tuple[int, str]]:
    """(size, path) of files created or changed since the task started."""
    wd = self.session.workdir or "/app"
    roots = " ".join(shlex.quote(r) for r in dict.fromkeys([wd, "/filesystem", "/app"]))
    res = await self.session.exec(
        f"for r in {roots}; do [ -d \"$r\" ] && find \"$r\" -xdev -type f -newermt"
        f" @{int(self.started)} -not -path '*/.git/*' -not -path '*/__pycache__/*'"
        " -not -name '*.pyc' -printf '%s\\t%p\\n' 2>/dev/null; done | sort -u -k2 | head -n 200",
        60)
    rows = [r.split("\t", 1) for r in (res.get("output") or "").strip().splitlines() if "\t" in r]
    return [(int(a), b) for a, b in rows if a.isdigit()]

  async def script_outputs(self, files: list[tuple[int, str]]) -> tuple[list[str], list[tuple[int, str]]]:
    """Runs delivered generator scripts the way a grader would; returns (problems, produced files)."""
    instr = self.session.instruction
    named = wanted_paths(instr, self.session.workdir or '')
    scripts = [p for _, p in files if p.endswith(".py") and (
        any(p == n or p.endswith("/" + n.lstrip("./")) for n in named) or "/output" in p or "/submission" in p)]
    if not scripts or "sys.argv" not in instr and "argv" not in instr:
      return [], []
    m = re.search(r"\.(ifc|xlsx|docx|dxf|json|csv|pdf|pptx)\b", instr, re.I)
    target_ext = "." + (m.group(1).lower() if m else "out")
    probs, produced = [], []
    for i, sc in enumerate(scripts[:3]):
      out = f"/tmp/_harness_check{i}/candidate{target_ext}"
      d = posixpath.dirname(out)
      res = await self.session.exec(
          f"rm -rf {d} && mkdir -p {d} && cd {shlex.quote(posixpath.dirname(sc))} && "
          f"( time timeout 600 python3 {shlex.quote(sc)} {out} ) > {d}/log.txt 2>&1; echo EXIT=$?; "
          f"tail -c 1500 {d}/log.txt; ls -la {d}", 700)
      text = res.get("output") or ""
      code = re.search(r"EXIT=(\d+)", text)
      if not code or code.group(1) != "0":
        probs.append(f"- {sc}: running it as the grader would (python3 {sc} <out>) failed:\n" + text[-1500:])
        continue
      listing = await self.session.exec(f"find {d} -type f -not -name log.txt -printf '%s\\t%p\\n'", 30)
      rows = [r.split("\t", 1) for r in (listing.get("output") or "").splitlines() if "\t" in r]
      if not rows:
        probs.append(f"- {sc}: ran but wrote nothing to the path given as sys.argv[1]")
      produced += [(int(a), b) for a, b in rows if a.isdigit()]
    return probs, produced

  def exemplar_for(self, path: str) -> str | None:
    """A same-format example among the task's input files, if any."""
    e = localtools.ext(path)
    same = [p for sz, p in self.inputs if localtools.ext(p) == e and p != path and sz < 20_000_000]
    named = [p for p in same if re.search(r"example|sample|template|reference|expected", p, re.I)]
    if named:
      return named[0]
    return same[0] if len(same) == 1 else None

  async def invalid_outputs(self) -> list[str]:
    """Format problems in the delivered files that a strict reader or grader would hit.

    Also fills self.measured: harness-side measurements of the structured deliverables.
    """
    files = await self.new_files()
    files = [(s, p) for s, p in files if not p.startswith(("/tmp/_harness_check", "/tmp/work", "/tmp/audit"))]
    probs, produced = await self.script_outputs(files)
    measured = []
    for size, path in files + produced:
      e = localtools.ext(path)
      if e not in (".ifc", ".json", ".csv", ".xlsx", ".xlsm", ".docx", ".pptx", ".pdf", ".dxf", ".zip"):
        continue
      if size > 40_000_000:
        continue
      try:
        self.forget(path)
        data = await self.fetch(path)
        issues = await asyncio.to_thread(localtools.validate, path, data)
        fact = await asyncio.wait_for(asyncio.to_thread(localtools.facts, path, data), 150)
        ex = self.exemplar_for(path)
        diff = []
        if ex:
          diff = await asyncio.to_thread(localtools.exemplar_diff, path, data, ex, await self.fetch(ex))
      except Exception as exc:  # pylint: disable=broad-except
        issues, fact, diff, ex = [f"could not read: {exc}"], "", [], None
      if issues:
        probs.append(f"- {path}:\n    " + "\n    ".join(issues))
      note = ""
      if fact:
        note += "\n  " + fact.replace("\n", "\n  ")
      if diff:
        note += (f"\n  structural differences from the task's own example {ex} (a grader built for the example"
                 " may break on these):\n    " + "\n    ".join(diff))
      if note:
        measured.append(f"* {path}:{note}")
    self.measured = "\n".join(measured)
    return probs

  async def pre_issue_note(self) -> str | None:
    """One-time note before the audit: unopened inputs and harness measurements of the outputs."""
    parts = []
    rest = [(sz, p) for sz, p in self.unopened()
            if not p.endswith((".pyc", ".DS_Store")) and "/." not in p]
    if rest and len(rest) <= 150:
      rows = []
      for sz, p in rest[:60]:
        rows.append(f"  {sz / 1024:8.1f} KB  {p}" + (f"   :: {self.preview_map[p]}" if p in self.preview_map else ""))
      parts.append(
          f"You have not opened {len(rest)} of the {len(self.inputs)} project files. Decisive facts often sit in"
          " files whose names look irrelevant (a calendar, an exhibit, a later letter, a checklist, a revision"
          " log). Open each one that could bear on the task (inspect/view/read_file) and correct the deliverables"
          " if anything changes; skip only what is clearly unrelated.\n" + "\n".join(rows)
          + ("\n  ..." if len(rest) > 60 else ""))
    if getattr(self, "measured", ""):
      parts.append(
          "The harness measured your output files the way a consumer would. Compare every line with what the"
          " task and inputs require (units, real-world size, counts, types, overlaps) and fix any mismatch:\n"
          + localtools.clip(self.measured, 12000))
    if not parts:
      return None
    return ("Pre-issue check before the independent review.\n\n" + "\n\n".join(parts)
            + "\n\nWhen done, answer with your updated final summary.")

  async def deliverable_dump(self) -> str:
    """Text of the files written during the task, for the reviewer."""
    wd = self.session.workdir or "/app"
    roots = " ".join(shlex.quote(r) for r in dict.fromkeys([wd, "/filesystem", "/app"]))
    res = await self.session.exec(
        f"for r in {roots}; do [ -d \"$r\" ] && find \"$r\" -xdev -type f -newermt"
        f" @{int(self.started)} -not -path '*/.git/*' -not -path '*/__pycache__/*'"
        " -not -name '*.pyc' -printf '%s\\t%p\\n' 2>/dev/null; done | sort -u -k2 | head -n 80",
        60)
    rows = [r.split("\t", 1) for r in (res.get("output") or "").strip().splitlines() if "\t" in r]
    named = set(wanted_paths(self.session.instruction, self.session.workdir or ''))
    rows.sort(key=lambda r: (0 if any(r[1].startswith(n) for n in named) else 1, r[1]))
    parts, budget = [], REVIEW_CHARS
    listing = "\n".join(f"{int(s) / 1024:8.1f} KB  {p}" for s, p in rows)
    parts.append("Files created or changed during the task:\n" + (listing or "(none)"))
    for size, path in rows:
      if budget <= 2000:
        parts.append(f"[{path}: not shown, review budget used]")
        continue
      e = localtools.ext(path)
      if e in (".py", ".sh") and not any(path.startswith(n) for n in named):
        continue  # helper scripts, not deliverables
      try:
        if int(size) > 25_000_000:
          text = "(large file, not shown)"
        else:
          self.forget(path)
          data = await self.fetch(path)
          if e in localtools.IMAGE_EXT:
            text = f"(image, {len(data)} bytes)"
          else:
            text = await asyncio.to_thread(localtools.inspect, path, data)
      except Exception as exc:  # pylint: disable=broad-except
        text = f"(could not read: {exc})"
      text = localtools.clip(text, min(budget, 40_000))
      budget -= len(text)
      parts.append(f"\n[[FILE {path}]]\n{text}")
    return "\n".join(parts)

  async def review(self, summary: str) -> str | None:
    """Independent check by a fresh-context agent with read-only tools; returns defects or None."""
    try:
      dump = await self.deliverable_dump()
    except taskbridge.TaskError as exc:
      await self.session.log(f"review: dump failed: {exc}")
      dump = "(could not list the delivered files)"
    rest = self.unopened()
    unopened = "\n".join(f"  {p}" for _, p in rest[:80]) or "  (none)"
    brief = (
        "THE TASK:\n" + self.session.instruction
        + "\n\nTHE COLLEAGUE'S SUMMARY:\n" + (summary or "(none)")
        + "\n\nHARNESS MEASUREMENTS OF THE DELIVERED FILES:\n" + (getattr(self, "measured", "") or "(none)")
        + f"\n\nINPUT FILES THE COLLEAGUE NEVER OPENED ({len(rest)} of {len(self.inputs)}):\n" + unopened
        + "\n\nTHE DELIVERED FILES (text extraction; open them yourself to check details):\n"
        + localtools.clip(dump, 60000))
    left = self.session.time_left()
    minutes = 30.0 if left is None else max(5.0, min(45.0, AUDIT_SHARE * left / 60))
    auditor = Auditor(self.session)
    auditor.adopt(self)
    try:
      text = await auditor.audit(brief, minutes, AUDIT_STEPS)
    except Exception as exc:  # pylint: disable=broad-except
      await self.session.log(f"review failed: {type(exc).__name__}: {str(exc)[:200]}")
      return None
    self.opened |= auditor.opened
    await self.session.log(f"review {self.reviews} ({auditor.used} tool calls): {(text or '')[:4000]}")
    if not text or "NO_ISSUES" in text[:40]:
      return None
    return text

  def minutes_left(self) -> float:
    left = self.session.time_left()
    return float("inf") if left is None else left / 60

  # ------------------------------------------------------------ loop

  async def run(self) -> str:
    import time  # pylint: disable=import-outside-toplevel

    self.started = time.time() - 5
    mcp = await self.connect_mcp()
    left = self.session.time_left()
    self.finish_margin = (
        serve.FINISH_MARGIN_SEC if left is None
        else min(serve.FINISH_MARGIN_SEC, max(left, 0) / 4))
    survey = await self.survey()
    self.messages = [
        {"role": "system", "content": serve.SYSTEM.format(
            workdir=self.session.workdir, mcp=mcp,
            minutes=int(left / 60) if left else "unlimited") + EXTRA_SYSTEM},
        {"role": "user", "content": self.session.instruction
         + "\n\n----- harness survey of the workspace -----\n" + survey},
    ]
    nudges, head = 0, ""
    while True:
      left = self.session.time_left()
      if left is not None and left < self.finish_margin:
        return await self.finish("Time is almost up.")
      try:
        reply = await self.complete(head=head)
      except OutOfTime:
        return await self.finish("Time is almost up, and the model is not answering.")
      except CutOff:
        return await self.finish("Time is almost up.")
      arguments = [parse_arguments(call) for call in reply.calls.values()]
      if reply.text or reply.calls:
        self.messages.append(reply.message())
      text = visible(head + reply.text)
      if text:
        self.session.final_message = text
      if not reply.calls:
        if visible(reply.text) and reply.finish == "length":
          nudges, head = 0, head + reply.text
          self.messages.append({"role": "user", "content": (
              f"Your reply was cut off at the {self.max_output}-token output limit. Continue where it"
              " stopped, and put long content in files with write_file rather than in a reply.")})
          continue
        if visible(reply.text) or nudges >= 2:
          follow = await self.after_answer(text or self.session.final_message)
          if follow is None:
            return text or self.session.final_message
          nudges, head = 0, ""
          self.messages.append({"role": "user", "content": follow})
          continue
        nudges += 1
        self.messages.append({"role": "user", "content": (
            "Continue the task; when it is done, answer with your summary of the deliverables.")})
        continue
      nudges, head = 0, ""
      for call, args in zip(reply.calls.values(), arguments):
        result = (await self.run_tool(call["name"], args)
                  if isinstance(args, dict) else args)
        if reply.finish == "length" and isinstance(args, str):
          result += (f" Your reply hit the {self.max_output}-token output limit; write"
                     " large files in parts.")
        self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
      left = self.session.time_left()
      if left is not None:
        self.messages[-1]["content"] += f"\n[{int(left // 60)} minutes left]"
      self.attach_images()
      if self.prompt_tokens > self.compact_at():
        await self.compact()

  async def after_answer(self, summary: str) -> str | None:
    """Runs the gate and the review; returns a follow-up prompt or None."""
    try:
      missing = await self.missing_outputs() if self.gate_checks < 3 else []
    except taskbridge.TaskError:
      missing = []
    if missing:
      self.gate_checks += 1
      await self.session.log(f"gate: missing {missing}")
      return ("These output paths named in the task do not exist yet: " + ", ".join(missing)
              + ". Create each one exactly there (or, if one is not actually a deliverable, say why),"
                " then answer with your summary.")
    if self.format_checks < 3:
      try:
        bad = await self.invalid_outputs()
      except Exception:  # pylint: disable=broad-except
        bad = []
      if bad:
        self.format_checks += 1
        await self.session.log(f"format gate: {bad}"[:1500])
        return ("A strict file validator (the kind graders use) found problems in your output files. Files that"
                " fail to open or validate usually score zero. Fix each one, preferably by generating the file"
                " with the format's standard library (e.g. ifcopenshell.api / ifcopenshell.guid.new() for IFC,"
                " openpyxl, python-docx, ezdxf) instead of hand-writing it, re-open it to verify, then answer"
                " with your summary.\n\n" + "\n".join(bad))
    if self.minutes_left() < REVIEW_MIN_LEFT_MIN:
      return None
    if self.coverage_checks < 1:
      self.coverage_checks += 1
      note = await self.pre_issue_note()
      if note:
        await self.session.log(f"pre-issue note: {note[:600]}")
        return note
    if self.reviews >= REVIEW_ROUNDS:
      return None
    self.reviews += 1
    defects = await self.review(summary)
    if not defects:
      return None
    return ("An independent senior engineer checked your deliverables against the input files and the task, with"
            " their own tools, and reported the issues below (with evidence). Verify each against the inputs; fix"
            " every valid one in the files themselves (the inputs win over both of you), re-run the relevant"
            " validation on the finished files, then answer with your updated final summary.\n\n" + defects)


class Auditor(EngineerAgent):
  """A fresh-context checker that shares the task session but must not change files."""

  def adopt(self, parent: EngineerAgent) -> None:
    """Takes over what the parent learned about the endpoint and the workspace."""
    for name in ("output_ceiling", "context_window", "max_output", "token_limit", "effort", "responses",
                 "finish_margin", "mcp", "inputs", "started"):
      if hasattr(parent, name):
        setattr(self, name, getattr(parent, name))
    self.preview_map = getattr(parent, "preview_map", {})
    self.tools = [t for t in parent.tools
                  if (t.get("function") or {}).get("name") not in ("write_file", "edit_file")]
    self.read_only = True
    self.used = 0

  async def audit(self, brief: str, minutes: float, steps: int) -> str | None:
    import time  # pylint: disable=import-outside-toplevel

    deadline = time.time() + minutes * 60
    self.messages = [
        {"role": "system", "content": AUDIT_SYSTEM.format(
            workdir=self.session.workdir, minutes=int(minutes), steps=steps)},
        {"role": "user", "content": brief},
    ]
    nudged = False
    while True:
      left = self.session.time_left()
      over = (self.used >= steps or time.time() > deadline
              or (left is not None and left < self.finish_margin + 600))
      try:
        if over:
          self.messages.append({"role": "user", "content": (
              "Stop investigating now. Answer with your verified findings (or exactly NO_ISSUES).")})
          reply = await self.complete(tool_choice="none", head="")
        else:
          reply = await self.complete(head="")
      except (OutOfTime, CutOff, openai.OpenAIError):
        return None
      if reply.text or reply.calls:
        self.messages.append(reply.message())
      if over or not reply.calls:
        text = visible(reply.text).strip()
        if text or over or nudged:
          return text or None
        nudged = True
        self.messages.append({"role": "user", "content": "Answer with your findings, or exactly NO_ISSUES."})
        continue
      arguments = [parse_arguments(call) for call in reply.calls.values()]
      for call, args in zip(reply.calls.values(), arguments):
        result = await self.run_tool(call["name"], args) if isinstance(args, dict) else args
        self.used += 1
        self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
      self.messages[-1]["content"] += (
          f"\n[{max(0, steps - self.used)} tool calls and {max(0, int((deadline - time.time()) // 60))} minutes"
          " left for this check]")
      self.attach_images()
      if self.prompt_tokens > self.compact_at():
        await self.compact()


async def solve(session: taskbridge.Session) -> str:
  print(f"task: workdir {session.workdir}, model {serve.MODEL},"
        f" {len(session.mcp_servers)} MCP server(s)", file=sys.stderr)
  return await EngineerAgent(session).run()


if __name__ == "__main__":
  taskbridge.run(solve)
