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
import fnmatch
import json
import os
import posixpath
import re
import shlex
import sys
import threading
import time
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
REVIEW_CHARS = int(os.environ.get("REVIEW_CHARS", "60000"))
VIEWS_PER_TURN = 8  # images one assistant turn may ask for
HEAVY_SEC = 240  # one harness-side parse or render may take this long
CACHE_BYTES = 300_000_000  # fetched files kept in memory
FETCH_MAX = 400_000_000  # larger files are not pulled out of the task container
# never listed, previewed or treated as deliverables: MCP server code and environments, hidden folders
PRUNE = (" -not -path '*/.git/*' -not -path '*/node_modules/*' -not -path '*/__pycache__/*'"
         " -not -path '/app/tools/*' -not -path '*/.venv/*' -not -path '*/site-packages/*'"
         " -not -path '*/.*/*' -not -name '*.pyc'")
FILE_EXT = {"md", "txt", "csv", "tsv", "json", "xml", "yaml", "yml", "xlsx", "xlsm", "xls", "docx", "pptx", "pdf",
            "dxf", "ifc", "zip", "py", "png", "jpg", "svg", "html", "geojson", "kml"}
_HEAVY = threading.Lock()  # pdfium and pyplot are not thread-safe; also keeps abandoned work from piling up


async def heavy(fn: Any, *args: Any, timeout: float = HEAVY_SEC) -> Any:
  """Runs a localtools call in a thread, one at a time, and gives up (TimeoutError) after `timeout` seconds."""
  gone = threading.Event()

  def work():
    with _HEAVY:
      if gone.is_set():
        raise TimeoutError("abandoned")
      return fn(*args)

  try:
    return await asyncio.wait_for(asyncio.to_thread(work), timeout)
  except (asyncio.TimeoutError, asyncio.CancelledError):
    gone.set()
    raise
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
- Keep a requirements register in your notes: each requirement you apply, with its source file, clause/section and \
value. Put it in a deliverable only where the task asks for the requirements applied, a basis or a register.
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
- Do not finish before every deliverable is complete and you have made a verification pass of its numbers, dates \
and verdicts against the input documents. When those checks pass, finish: do not rework content that is correct.

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
{workdir}), read_file, inspect and view{mcp}. Do NOT modify or delete any project input or deliverable, and never call a \
tool that sends, creates, updates or deletes anything; write scratch files only under /tmp/audit. You have about \
{minutes} minutes and {steps} tool calls.

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


AUDIT_NOTES_PROMPT = """The conversation is getting long, so the earlier turns will be removed. Write notes that let \
you finish this independent check without them: every defect you have verified so far (file, location, evidence, \
fix), what you checked and found correct, and what remains to check. Be specific."""

FINAL_SUMMARY_PROMPT = """Now answer with your complete final summary, written for the client as if it were your \
first answer: every deliverable with its path, the key results and numbers with units, the verdicts or decisions, \
and any open item the client must know. Do not mention reviews, checks, the harness, or what you changed along the \
way. Do not call tools."""

AGAIN = (" Then answer with your complete final summary again (every deliverable path and the key results), written"
         " for the client: it replaces your previous answer, so do not refer to it or to this check.")
CHATTER_RE = re.compile(
    r"\b(fixed (the|both|all|it|this|that)|(both|all|the) fixes|fixes (are|were|have)|the reviewer"
    r"|independent (check|review|engineer)|pre-issue|as before|previous (summary|answer)|no (further )?changes"
    r"|updated (final )?summary|issues? (fixed|addressed|resolved)|(is|are) now (fixed|corrected))\b",
    re.I)
READ_TOOL_RE = re.compile(
    r"(^|[_-])(read|get|list|search|find|query|view|show|fetch|describe|count|info|lookup|preview|export|stat)"
    r"([_-]|$)", re.I)
WRITE_TOOL_RE = re.compile(
    r"(^|[_-])(write|create|update|delete|remove|send|reply|forward|add|insert|set|edit|move|rename|upload|save"
    r"|put|post|patch|append|modify|replace|draft|schedule|cancel|accept|decline|execute|run|import)([_-]|$)", re.I)


def issue_key(issue: str) -> str:
  """A validator finding without its count and example instance."""
  return re.sub(r" \(\d+x, e\.g\. .*$", "", issue, flags=re.S)


def wanted_paths(instruction: str, workdir: str = "", known_dirs: Any = ()) -> list[str]:
  """Paths the instruction names (likely deliverables or inputs).

  Relative ones (with a folder part and a file extension) are resolved against the folder the instruction says the
  work happens in, the workdir, or the folder of a named absolute path: the first whose parent folder is known to
  exist (known_dirs). Bare names, numbers and identifiers in backticks are not paths.
  """
  seen: list[str] = []
  for m in PATH_RE.finditer(instruction):
    tail = re.match(r"[^\s`'\"),;]*", instruction[m.end(1):]).group(0)
    if any(ch in tail for ch in "<{*[$%"):
      continue  # cut at a placeholder or glob: not a real path
    p = m.group(1).rstrip(".,;:)")
    if p not in seen:
      seen.append(p)
  if workdir:
    absolute = list(seen)
    said = re.search(r"\bworking (?:in|under|from|inside) `?(/[\w./@+-]*[\w/])`?", instruction)
    bases = ([said.group(1)] if said else []) + [workdir]
    for a in absolute:  # folders named absolutely, and their parents ("<task dir>/output/x" -> "<task dir>")
      d = a.rstrip("/") if posixpath.splitext(a)[1] == "" else posixpath.dirname(a)
      for cand in (d, posixpath.dirname(d)):
        if cand not in bases and cand not in ("", "/"):
          bases.append(cand)
    known = set(known_dirs or ())
    for m in REL_RE.finditer(instruction):
      p = m.group(1)
      rel = p[2:] if p.startswith("./") else p
      if (p.startswith("/") or "/" not in rel or rel.startswith(".")
          or rel.rsplit(".", 1)[-1].lower() not in FILE_EXT
          or any(a.endswith("/" + rel) for a in absolute)):
        continue
      def exists(b: str) -> bool:
        d = posixpath.dirname(posixpath.normpath(posixpath.join(b, rel)))
        return d in known or posixpath.dirname(d) in known
      base = next((b for b in bases if exists(b)), bases[0])
      p = posixpath.normpath(posixpath.join(base, rel))
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
    self.errors: list[str] = []
    self.edits: dict[str, int] = {}
    self.calls = 0
    self.preview_map: dict[str, str] = {}
    self.measured = ""
    self.started = time.time() - 5
    self.aliases: list[tuple[str, str]] = []  # (symlink, target) of the project roots
    self.asked_missing: set[str] = set()
    self.script_ok: set[tuple[str, int, str]] = set()
    self.file_mtime: dict[str, str] = {}
    self.baseline: dict[str, dict[str, str]] = {}  # ext -> validator finding -> input file that has it too
    self.best_summary = ""
    self.followups = 0
    self.summary_asked = False
    self.dropped_views = 0
    self.past_findings: list[str] = []
    self.audit_opened: set[str] = set()  # read by the checker, not by the engineer

  # ------------------------------------------------------------ setup

  def roots(self) -> list[str]:
    """Project roots, most specific first, each ending in '/' (so that a symlinked root is followed)."""
    wd = self.session.workdir or "/app"
    return [r.rstrip("/") + "/" for r in dict.fromkeys([wd, "/filesystem", "/app"])]

  def find_cmd(self, extra: str, cap: int) -> str:
    """A shell pipeline printing 'size<TAB>mtime<TAB>path' of project files, capped per root."""
    roots = " ".join(shlex.quote(r) for r in self.roots())
    return (f"for r in {roots}; do [ -d \"$r\" ] && find \"$r\" -xdev -type f{PRUNE}{extra}"
            f" -printf '%s\\t%T@\\t%p\\n' 2>/dev/null | sort -t\"$(printf '\\t')\" -k3 | head -n {cap}; done")

  @staticmethod
  def rows(output: str, cap: int) -> list[tuple[int, str, str]]:
    """(size, mtime, path) rows of find_cmd output, without duplicates."""
    out, seen = [], set()
    for row in (output or "").splitlines():
      parts = row.split("\t", 2)
      if len(parts) != 3 or not parts[0].isdigit():
        continue
      path = posixpath.normpath(parts[2])
      if path not in seen:
        seen.add(path)
        out.append((int(parts[0]), parts[1], path))
    return out[:cap]

  def named(self) -> list[str]:
    """Paths the task names, relative ones resolved with what the survey knows of the folders."""
    dirs = set()
    for _, p in self.inputs:
      d = posixpath.dirname(p)
      while d and d != "/" and d not in dirs:
        dirs.add(d)
        d = posixpath.dirname(d)
    try:
      return wanted_paths(self.session.instruction, self.session.workdir or "", dirs)
    except Exception:  # pylint: disable=broad-except
      return []

  async def survey(self) -> str:
    """A listing of the project files and the container's tooling."""
    cmd = (
        self.find_cmd("", SURVEY_FILES) + "; echo '@@H@@';"
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
        " >/dev/null && printf '%s ' $t; done; echo; echo '@@H@@';"
        " for l in /filesystem /app/files /workspace /app; do [ -L $l ] && echo \"$l $(readlink -f $l)\"; done; true"
    )
    try:
      res = await self.session.exec(cmd, 90)
      out = res.get("output") or ""
    except taskbridge.TaskError as exc:
      return f"(survey failed: {exc})"
    files, tooling, links = (out.split("@@H@@") + ["", ""])[:3]
    for row in links.split("\n"):
      link, _, target = row.strip().partition(" ")
      if link.startswith("/") and target.startswith("/") and link != target:
        self.aliases.append((link, target.rstrip("/") or "/"))
    lines = []
    for size, _, path in self.rows(files, SURVEY_FILES):
      lines.append((path, f"{size / 1024:9.1f} KB  {path}"))
      self.inputs.append((size, path))
    if not lines:
      await self.session.log("survey: no project files found")
    named = self.named()
    named_note = ""
    if named:
      named_note = "\nPaths named in the task: " + ", ".join(named)
    try:
      await asyncio.wait_for(self.previews(), 240)
    except Exception as exc:  # pylint: disable=broad-except
      await self.session.log(f"survey: previews stopped: {type(exc).__name__}: {str(exc)[:200]}")
    previews = self.preview_map  # whatever was gathered, also after a timeout
    lines = [ln + ("   :: " + previews[p] if p in previews else "") for p, ln in lines]
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
    out = self.preview_map  # filled as it goes, so a timeout keeps what was done
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
        out[p] = await heavy(localtools.preview, p, data, timeout=30)
      except taskbridge.Cancelled:
        raise
      except Exception:  # pylint: disable=broad-except
        continue
    return out

  def unopened(self) -> list[tuple[int, str]]:
    """Input files no tool call appears to have touched."""
    return [(sz, p) for sz, p in self.inputs if p not in self.opened and sz > 0]

  def canon(self, path: str, here: str | None = None) -> str:
    """An absolute, normalised path with the known symlinked roots resolved."""
    path = path if path.startswith("/") else posixpath.join(here or self.session.workdir or "/", path)
    path = posixpath.normpath(path)
    for link, target in self.aliases:
      if path == link or path.startswith(link + "/"):
        return posixpath.normpath(target + path[len(link):])
    return path

  def note_opened(self, name: str, args: dict[str, Any]) -> None:
    """Records which input files a tool call read. Generous: a false 'opened' costs less than a false 'unopened'."""
    try:
      if name in ("read_file", "inspect", "view"):
        self.opened.add(self.canon(str(args.get("path", ""))))
        return
      if name == "bash":
        text = str(args.get("command", ""))
      elif name == "write_file":  # a script that will read files
        text = str(args.get("content", ""))
      elif name == "edit_file":
        text = str(args.get("new_text", ""))
      else:  # the task's own tools
        text = json.dumps(args, ensure_ascii=False)
      if not text or all(p in self.opened for _, p in self.inputs):
        return
      heres = [self.canon(".")] + [self.canon(m.group(1).strip("'\""))
                                    for m in re.finditer(r"\bcd\s+([^\s;&|)]+)", text)]
      exact, dirs, globs = set(), set(), []
      lookers = {"cd", "ls", "tree", "mkdir", "du", "stat", "pwd", "echo", "test", "[", "rm", "cp", "mv", "touch"}
      for seg in re.split(r"[;&|\n]+", text):
        toks = [t.strip("'\"`,:=()[]{}") for t in re.findall(r"[^\s<>]+", seg)]
        toks = [t for t in toks if t]
        if not toks:
          continue
        listing = toks[0] in lookers or (toks[0] == "find" and "-exec" not in seg and "xargs" not in text)
        for t in toks:
          t = t.split("=", 1)[-1].strip("'\"") or t
          for here in heres:
            c = self.canon(t, here)
            if any(ch in t for ch in "*?"):
              globs.append(c)
            else:
              exact.add(c)
              if not listing and c not in ("/", "/app", "/filesystem", "/tmp", "/app/files"):
                dirs.add(c)
      dirs -= set(heres[1:])  # a bare `cd dir` reads nothing
      names = set(re.findall(r"[\w.@+-]+", text))
      for _, p in self.inputs:
        if p in self.opened:
          continue
        base = posixpath.basename(p)
        hit = p in text or p in exact or base in names or (len(base) >= 6 and base in text)
        if not hit:
          hit = any(fnmatch.fnmatch(p, g) for g in globs)
        if not hit:
          d = posixpath.dirname(p)
          while d and d != "/":
            if d in dirs:
              hit = True
              break
            d = posixpath.dirname(d)
        if hit:
          self.opened.add(p)
    except Exception:  # pylint: disable=broad-except
      pass

  async def read_file(self, path: str) -> str:
    out = await super().read_file(path)
    e = localtools.ext(path)
    if " is a binary file (" in out[:400] and (e in DOC_EXT or e in localtools.IMAGE_EXT):
      return (f"{path} is a binary {e} file; showing the inspect tool's extraction instead"
              " (use inspect for pages/sheets/queries, view to look at it):\n" + await self.tool_inspect(path))
    return out

  # ------------------------------------------------------------ files

  async def fetch(self, path: str) -> bytes:
    if not isinstance(path, str) or not path:
      raise taskbridge.TaskError("path must be a non-empty string")
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
      size = result.get("size")
      if isinstance(size, (int, float)) and size > FETCH_MAX:
        raise taskbridge.TaskError(
            f"{path} is {int(size) >> 20} MB, too large for the harness tools; work on it with bash in the container")
      data = await self.fetch_parts(path)
    if len(data) <= 64_000_000:
      while self.cache and (len(self.cache) > 40
                            or sum(len(v) for v in self.cache.values()) + len(data) > CACHE_BYTES):
        self.cache.pop(next(iter(self.cache)))
      self.cache[path] = data
    return data

  async def fetch_parts(self, path: str) -> bytes:
    """Reads a file over 10 MB by splitting it in the container."""
    tmp = "/tmp/.harness_parts"
    q = shlex.quote(path)
    res = await self.session.exec(
        f"rm -rf {tmp} && mkdir -p {tmp} && split -b 8M -d -a 3 {q} {tmp}/p_ && ls {tmp}", 120)
    parts = sorted(p for p in (res.get("output") or "").split() if re.fullmatch(r"p_\d{3}", p))
    if res.get("exit_code") not in (0, None) or not parts:
      raise taskbridge.TaskError(f"cannot read {path} in parts: {(res.get('output') or '')[-200:]}")
    chunks = []
    try:
      for p in parts:
        chunks.append(await self.session.read_file(f"{tmp}/{p}"))
    finally:
      try:
        await self.session.exec(f"rm -rf {tmp}", 30)
      except taskbridge.TaskError:
        pass
    return b"".join(chunks)

  def forget(self, path: str) -> None:
    self.cache.pop(self.path(path), None)

  # ------------------------------------------------------------ tools

  async def run_tool(self, name: str, args: dict[str, Any]) -> str:
    """Never raises (except for cancellation): a bad call comes back as an error string."""
    try:
      return await self.run_tool_inner(name, args)
    except taskbridge.Cancelled:
      raise
    except Exception as exc:  # pylint: disable=broad-except
      return f"error: {type(exc).__name__}: {str(exc)[:500]}"

  async def run_tool_inner(self, name: str, args: dict[str, Any]) -> str:
    if not isinstance(args, dict):
      return "error: the arguments must be a JSON object"
    if self.read_only and (name in ("write_file", "edit_file") or name in self.blocked_tools):
      return "error: the checker must not change anything; report the defect and its fix instead"
    self.note_opened(name, args)
    if name not in ("read_file", "inspect", "view", "mcp_find"):
      # a write, a command or a task tool may change any cached file
      self.cache.clear()
    if name == "inspect":
      return await self.tool_inspect(
          **{k: v for k, v in args.items() if k in ("path", "pages", "sheet", "cells", "query")})
    if name == "view":
      return await self.tool_view(**{k: v for k, v in args.items() if k in ("path", "page", "crop", "dpi")})
    if self.read_only and name == "mcp_call" and str(args.get("tool", "")) in self.blocked_tools:
      return "error: the checker must not change anything; report the defect and its fix instead"
    result = await super().run_tool(name, args)
    try:
      return result + self.governor(name, args, result)
    except Exception:  # pylint: disable=broad-except
      return result

  blocked_tools: frozenset[str] = frozenset()

  def governor(self, name: str, args: dict[str, Any], result: str) -> str:
    """Short advice appended to a tool result when the work shows signs of churn."""
    note = ""
    failed = result.startswith("error:") or (
        result.startswith("exit code") and not result.startswith("exit code 0")) or "Traceback (most recent" in result
    if failed:
      lines = [ln.strip() for ln in result.splitlines() if ln.strip() and not ln.startswith("[")]
      sig = re.sub(r"\d+", "#", lines[-1])[:160] if lines else "error"
      self.errors.append(sig)
      recent = self.errors[-8:]
      if recent.count(sig) in (3, 6):
        note += ("\n[harness: this same error has now come back " + str(recent.count(sig)) + " times. Stop trying"
                 " variations: read the real interface (python3 -c 'help(...)', --help, the file itself), or take a"
                 " different approach. Do not skip the step or report it as passed.]")
    else:
      self.errors.append("")
    if name in ("write_file", "edit_file"):
      path = self.path(str(args.get("path", "")))
      n = self.edits[path] = self.edits.get(path, 0) + 1
      if n in (30, 60, 100):
        note += (f"\n[harness: {n} writes to {path}. Before more local patches, measure what the finished output"
                 " really contains (inspect it; check units, scale, counts against the inputs) and re-read your"
                 " notes: repeated patching usually means a wrong assumption upstream.]")
    self.calls += 1
    if self.calls % 150 == 0:
      note += (f"\n[harness: {self.calls} tool calls so far. Update /tmp/work/NOTES.md, re-check the task's"
               " requirements list, and make sure a complete, valid version of every deliverable is on disk now.]")
    return note

  async def tool_inspect(self, path: str = "", pages: Any = None, sheet: Any = None,
                         cells: Any = None, query: Any = None, **_: Any) -> str:
    await self.session.log(f"inspect {path} pages={pages} sheet={sheet} query={query}")
    try:
      data = await self.fetch(path)
      return await heavy(localtools.inspect, path, data, pages, sheet, cells, query)
    except taskbridge.Cancelled:
      raise
    except taskbridge.TaskError as exc:
      return f"error: {exc}"
    except asyncio.TimeoutError:
      return (f"error: inspecting {path} took over {HEAVY_SEC} s and was stopped; ask for less (pages, a sheet and"
              " cells range, a query) or work on the file with bash in the container")
    except Exception as exc:  # pylint: disable=broad-except
      return f"error: could not inspect {path}: {type(exc).__name__}: {str(exc)[:500]}"

  async def tool_view(self, path: str = "", page: Any = None, crop: Any = None,
                      dpi: Any = None, **_: Any) -> str:
    await self.session.log(f"view {path} page={page} crop={crop}")
    if len(self.pending_images) >= VIEWS_PER_TURN:
      self.dropped_views += 1
      return (f"error: not rendered: only {VIEWS_PER_TURN} views fit in one turn. Look at the {VIEWS_PER_TURN}"
              " images that follow, then ask for this one again.")
    try:
      data = await self.fetch(path)
      img = await heavy(localtools.render, path, data, page, crop, dpi)
    except taskbridge.Cancelled:
      raise
    except taskbridge.TaskError as exc:
      return f"error: {exc}"
    except asyncio.TimeoutError:
      return f"error: rendering {path} took over {HEAVY_SEC} s and was stopped; try a crop or a lower dpi"
    except Exception as exc:  # pylint: disable=broad-except
      return f"error: could not render {path}: {type(exc).__name__}: {str(exc)[:500]}"
    label = f"{path}" + (f" page {page}" if page else "") + (f" crop {crop}" if crop else "")
    self.pending_images.append((label, img))
    return f"rendered {label}; the image follows this result."

  def attach_images(self) -> None:
    if not self.pending_images:
      return
    note = ""
    if self.dropped_views:
      note = (f" ({self.dropped_views} more view call(s) this turn were not rendered: at most {VIEWS_PER_TURN}"
              " per turn; ask for them again)")
      self.dropped_views = 0
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Images you asked to view: "
         + "; ".join(lbl for lbl, _ in self.pending_images) + note}
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
    for n, m in enumerate(reversed(list(self.image_msgs))):
      imgs = [c for c in m["content"] if c.get("type") == "image_url"]
      if n == 0:  # the turn just made is always shown whole; only older turns are pruned
        count += len(imgs)
      elif count + len(imgs) > KEEP_IMAGES:
        text = m["content"][0]["text"]
        m["content"] = text + " [image removed to save context; view it again if needed]"
        self.image_msgs.remove(m)
      else:
        count += len(imgs)

  def strip_images(self) -> bool:
    """Replaces every image in the conversation by its label (after an endpoint refused the request)."""
    changed = False
    for m in self.messages:
      if isinstance(m.get("content"), list) and any(
          isinstance(c, dict) and c.get("type") == "image_url" for c in m["content"]):
        text = " ".join(str(c.get("text", "")) for c in m["content"] if isinstance(c, dict) and c.get("type") == "text")
        m["content"] = text + " [the model endpoint refused these images; use inspect, or view a tighter crop]"
        changed = True
    self.image_msgs = []
    return changed

  async def complete_soft(self, **kwargs: Any) -> serve.Reply:
    """complete(), retried once without images when the endpoint refuses the request outright."""
    try:
      return await self.complete(**kwargs)
    except openai.APIStatusError as exc:
      if exc.status_code in (400, 413, 422) and self.strip_images():
        await self.session.log(f"model call refused ({str(exc)[:200]}); retrying without images")
        return await self.complete(**kwargs)
      raise
    except serve.StreamBadRequest as exc:
      if self.strip_images():
        await self.session.log(f"model call refused ({str(exc)[:200]}); retrying without images")
        return await self.complete(**kwargs)
      raise

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
    named = [p for p in self.named() if p not in self.asked_missing]  # each path is asked for once
    if not named:
      return []
    quoted = " ".join(shlex.quote(p) for p in named)
    res = await self.session.exec(
        f"for p in {quoted}; do [ -e \"$p\" ] || echo \"$p\"; done", 30)
    gone = [p for p in (res.get("output") or "").split("\n") if p in named]
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
    res = await self.session.exec(self.find_cmd(f" -newermt @{int(self.started)}", 200), 60)
    rows = self.rows(res.get("output") or "", 400)
    for _, mtime, path in rows:
      self.file_mtime[path] = mtime
    return [(size, path) for size, _, path in rows]

  async def script_outputs(self, files: list[tuple[int, str]]) -> tuple[list[str], list[tuple[int, str]]]:
    """Runs a delivered script that the task says will be run with an output path; (problems, produced files).

    Only a script the task names is run, from an empty scratch folder (as a grader's sandbox would), within the
    time the task has left, and not again while it is unchanged since it last passed.
    """
    instr = self.session.instruction
    if "argv" not in instr:
      return [], []
    said = {n.lstrip("./") for n in re.findall(r"[\w./@+-]*\w\.py\b", instr)}
    named = set(self.named())
    scripts = [(s, p) for s, p in files if p.endswith(".py") and (
        p in named or any(p.endswith("/" + n) for n in said if n))]
    if not scripts:
      return [], []
    left = self.session.time_left()
    cap = 600 if left is None else int(min(600, left - self.finish_margin - 120))
    cap = min(cap, int(self.session.limits.get("exec_timeout_sec") or 600))
    if cap < 60:
      return [], []
    m = re.search(r"\.(ifc|xlsx|docx|dxf|json|csv|pdf|pptx)\b", instr, re.I)
    target_ext = "." + (m.group(1).lower() if m else "out")
    probs, produced = [], []
    for i, (size, sc) in enumerate(scripts[:2]):
      key = (sc, size, self.file_mtime.get(sc, ""))
      if key in self.script_ok:
        continue
      d = f"/tmp/_harness_check{i}"
      out = f"{d}/out/candidate{target_ext}"
      res = await self.session.exec(
          f"rm -rf {d} && mkdir -p {d}/cwd {d}/out && cd {d}/cwd && "
          f"( time PYTHONDONTWRITEBYTECODE=1 timeout {max(30, cap - 20)} python3 {shlex.quote(sc)} {out} )"
          f" > {d}/log.txt 2>&1; echo EXIT=$?; tail -c 1500 {d}/log.txt", cap)
      text = res.get("output") or ""
      code = re.search(r"EXIT=(\d+)", text)
      if res.get("timed_out") or (code and code.group(1) == "124"):
        probs.append(f"- advisory: {sc} did not finish within {cap - 20} s when run as `python3 {sc} <output path>`"
                     " from an empty folder; check the task's time limit for it")
        continue
      if not code or code.group(1) != "0":
        probs.append(f"- {sc}: running it as `python3 {sc} <output path>` from an empty scratch folder (as a"
                     " grader's sandbox would) failed. If it depends on the current folder, use paths relative"
                     " to the script (__file__) or absolute ones:\n" + text[-1500:])
        continue
      listing = await self.session.exec(f"find {d}/out -type f -printf '%s\\t%p\\n'", 30)
      rows = [r.split("\t", 1) for r in (listing.get("output") or "").splitlines() if "\t" in r]
      if not rows:
        probs.append(f"- {sc}: ran but wrote nothing to the path given as sys.argv[1]")
      else:
        self.script_ok.add(key)
      produced += [(int(a), b) for a, b in rows if a.isdigit()]
    return probs, produced

  def exemplar_for(self, path: str) -> str | None:
    """A same-format input whose file name says it is an example of the deliverable, if any."""
    e = localtools.ext(path)
    same = [p for sz, p in self.inputs if localtools.ext(p) == e and p != path and sz < 20_000_000]
    named = [p for p in same
             if re.search(r"example|sample|template|reference|expected", posixpath.basename(p), re.I)]
    return named[0] if named else None

  async def tolerated(self, e: str) -> dict[str, str]:
    """Validator findings that the task's own input files of this format show too (finding -> input file)."""
    if e in self.baseline:
      return self.baseline[e]
    found: dict[str, str] = {}
    self.baseline[e] = found
    if e != ".ifc":
      return found
    same = sorted((sz, p) for sz, p in self.inputs if localtools.ext(p) == e and 0 < sz <= localtools.IFC_HEAVY_BYTES)
    for _, p in same[:3]:
      try:
        for issue in await heavy(localtools.validate, p, await self.fetch(p), timeout=120):
          found.setdefault(issue_key(issue), p)
      except taskbridge.Cancelled:
        raise
      except Exception:  # pylint: disable=broad-except
        continue
    return found

  async def invalid_outputs(self) -> list[str]:
    """Format problems in the delivered files that a strict reader or grader would hit.

    Also fills self.measured: harness-side measurements of the structured deliverables.
    """
    files = await self.new_files()
    files = [(s, p) for s, p in files if not p.startswith(("/tmp/_harness_check", "/tmp/work", "/tmp/audit"))]
    try:
      probs, produced = await self.script_outputs(files)
    except taskbridge.Cancelled:
      raise
    except Exception as exc:  # pylint: disable=broad-except
      await self.session.log(f"format gate: script run failed in the harness: {type(exc).__name__}: {exc}"[:300])
      probs, produced = [], []
    measured = ["* " + p[2:] for p in probs if p.startswith("- advisory:")]
    probs = [p for p in probs if not p.startswith("- advisory:")]
    t0 = time.time()
    for size, path in files + produced:
      e = localtools.ext(path)
      if e not in (".ifc", ".json", ".csv", ".xlsx", ".xlsm", ".docx", ".pptx", ".pdf", ".dxf", ".zip"):
        continue
      if size > 40_000_000 or time.time() - t0 > 420:
        continue
      left = self.session.time_left()
      if left is not None and left < self.finish_margin + 120:
        break
      # Only what a reader of the FILE reports is a problem. A harness-side failure (fetch, timeout, crash of a
      # measurement) is logged and never shown as a defect of the file.
      try:
        self.forget(path)
        data = await self.fetch(path)
      except taskbridge.Cancelled:
        raise
      except Exception as exc:  # pylint: disable=broad-except
        await self.session.log(f"format gate: cannot fetch {path}: {type(exc).__name__}: {exc}"[:300])
        continue
      issues, notes, fact, diff, ex = [], [], "", [], None
      try:
        found = await heavy(localtools.validate, path, data, timeout=180)
        known = await self.tolerated(e) if found else {}
        for issue in found:
          if issue.startswith("advisory:"):
            notes.append(issue)
          elif issue_key(issue) in known:
            notes.append(f"{issue}  [also present in the task's own input {known[issue_key(issue)]}, so probably"
                         " tolerated]")
          else:
            issues.append(issue)
      except taskbridge.Cancelled:
        raise
      except Exception as exc:  # pylint: disable=broad-except
        await self.session.log(f"format gate: validator did not finish on {path}: {type(exc).__name__}: {exc}"[:300])
        notes.append("[the harness's validator did not finish on this file; validate it yourself]")
      try:
        fact = await heavy(localtools.facts, path, data, timeout=150)
        ex = self.exemplar_for(path)
        if ex:
          diff = await heavy(localtools.exemplar_diff, path, data, ex, await self.fetch(ex), timeout=120)
      except taskbridge.Cancelled:
        raise
      except Exception as exc:  # pylint: disable=broad-except
        await self.session.log(f"format gate: could not measure {path}: {type(exc).__name__}: {exc}"[:300])
      if issues:
        probs.append(f"- {path}:\n    " + "\n    ".join(issues))
      note = ""
      if notes:
        note += "\n  " + "\n  ".join(notes)
      if fact:
        note += "\n  " + fact.replace("\n", "\n  ")
      if diff:
        note += (f"\n  structural differences from the task's example file {ex} (check them against the task; a"
                 " grader built for the example may depend on its structure):\n    " + "\n    ".join(diff))
      if note:
        measured.append(f"* {path}:{note}")
    self.measured = "\n".join(measured)
    try:
      await self.session.exec("rm -rf /tmp/_harness_check[0-9]", 30)
    except taskbridge.TaskError:
      pass
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
          f"{len(rest)} of the {len(self.inputs)} project files were apparently not opened (the harness saw no"
          " tool call naming them; it cannot see reads made inside your scripts, so ignore any you did read)."
          " Decisive facts often sit in files whose names look irrelevant (a calendar, an exhibit, a later"
          " letter, a checklist, a revision log). Open each one that could bear on the task and that you have"
          " not read (inspect/view/read_file), and correct the deliverables only if something in it changes"
          " the answer; skip what is clearly unrelated.\n" + "\n".join(rows)
          + ("\n  ..." if len(rest) > 60 else ""))
    elif len(rest) > 150:
      await self.session.log(f"pre-issue note: {len(rest)} files apparently unopened; list not shown")
    if self.measured:
      parts.append(
          "The harness measured your output files the way a consumer would. These are measurements, not"
          " defects: compare each line with what the task and the inputs require (units, real-world size,"
          " counts, types) and change a file only where a line contradicts them:\n"
          + localtools.clip(self.measured, 12000))
    if not parts:
      return None
    return ("Pre-issue check before the independent review.\n\n" + "\n\n".join(parts)
            + "\n\nWhen done, leave correct work as it is." + AGAIN)

  async def deliverable_dump(self) -> str:
    """Text of the files written during the task, for the reviewer."""
    res = await self.session.exec(self.find_cmd(f" -newermt @{int(self.started)}", 80), 60)
    rows = [(size, path) for size, _, path in self.rows(res.get("output") or "", 80)]
    named = set(self.named())
    rows.sort(key=lambda r: (0 if any(r[1].startswith(n) for n in named) else 1, r[1]))
    parts, budget = [], REVIEW_CHARS
    listing = "\n".join(f"{s / 1024:8.1f} KB  {p}" for s, p in rows)
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
            text = await heavy(localtools.inspect, path, data, timeout=120)
      except taskbridge.Cancelled:
        raise
      except Exception as exc:  # pylint: disable=broad-except
        text = f"(the harness could not extract this file: {type(exc).__name__}; open it yourself)"
      text = localtools.clip(text, min(budget, 40_000))
      budget -= len(text)
      parts.append(f"\n[[FILE {path}]]\n{text}")
    return "\n".join(parts)

  async def review(self, summary: str) -> str | None:
    """Independent check by a fresh-context agent with read-only tools; returns defects or None."""
    final = self.session.final_message  # the auditor must never change what the grader receives
    try:
      return await self.review_inner(summary)
    except taskbridge.Cancelled:
      raise
    except Exception as exc:  # pylint: disable=broad-except
      await self.session.log(f"review failed: {type(exc).__name__}: {str(exc)[:200]}")
      return None
    finally:
      self.session.final_message = final

  async def review_inner(self, summary: str) -> str | None:
    try:
      dump = await self.deliverable_dump()
    except taskbridge.Cancelled:
      raise
    except Exception as exc:  # pylint: disable=broad-except
      await self.session.log(f"review: dump failed: {type(exc).__name__}: {exc}"[:300])
      dump = "(could not list the delivered files)"
    rest = [(sz, p) for sz, p in self.unopened() if p not in self.audit_opened]
    unopened = "\n".join(f"  {p}" for _, p in rest[:80]) or "  (none)"
    earlier = ""
    if self.past_findings:
      earlier = ("\n\nAN EARLIER CHECK ALREADY RAISED THE POINTS BELOW, and the colleague has responded to them in"
                 " the files and the summary above. Do not repeat a point unless you verify on the current files"
                 " that it still stands:\n" + localtools.clip("\n---\n".join(self.past_findings), 8000))
    brief = (
        "THE TASK:\n" + self.session.instruction
        + "\n\nTHE COLLEAGUE'S SUMMARY:\n" + (summary or "(none)")
        + "\n\nHARNESS MEASUREMENTS OF THE DELIVERED FILES (measurements, not defects):\n"
        + (localtools.clip(self.measured, 12000) or "(none)")
        + f"\n\nINPUT FILES THE COLLEAGUE APPARENTLY NEVER OPENED ({len(rest)} of {len(self.inputs)}; the list is"
          " approximate, reads inside scripts are not seen):\n" + unopened
        + earlier
        + "\n\nTHE DELIVERED FILES (text extraction; open them yourself to check details):\n" + dump)
    left = self.session.time_left()
    minutes = 30.0 if left is None else max(5.0, min(45.0, AUDIT_SHARE * left / 60))
    auditor = Auditor(self.session)
    auditor.adopt(self)
    try:
      text = await auditor.audit(brief, minutes, AUDIT_STEPS)
    except taskbridge.Cancelled:
      raise
    except Exception as exc:  # pylint: disable=broad-except
      await self.session.log(f"review failed: {type(exc).__name__}: {str(exc)[:200]}")
      return None
    self.audit_opened |= auditor.opened
    await self.session.log(f"review {self.reviews} ({auditor.used} tool calls): {(text or '')[:4000]}")
    if not text:
      return None
    bare = text.strip().strip("`*_. \n")
    if "NO_ISSUES" in text[:40] or bare.endswith("NO_ISSUES") and len(bare) < 400:
      return None
    self.past_findings.append(text)
    return text

  def minutes_left(self) -> float:
    left = self.session.time_left()
    return float("inf") if left is None else left / 60

  # ------------------------------------------------------------ loop

  def is_chatter(self, text: str) -> bool:
    """Whether a final reply reads as a note about fixes rather than a summary of the deliverables."""
    if not text:
      return True
    if len(text) < 0.6 * len(self.best_summary):
      return True
    return bool(CHATTER_RE.search(text[:700]))

  async def run(self) -> str:
    self.started = time.time() - 5
    mcp = await self.connect_mcp()
    left = self.session.time_left()
    self.finish_margin = (
        serve.FINISH_MARGIN_SEC if left is None
        else min(serve.FINISH_MARGIN_SEC, max(left, 0) / 4))
    try:
      survey = await self.survey()
    except taskbridge.Cancelled:
      raise
    except Exception as exc:  # pylint: disable=broad-except
      await self.session.log(f"survey failed: {type(exc).__name__}: {str(exc)[:200]}")
      survey = "(the survey of the workspace failed; list the files yourself)"
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
        reply = await self.complete_soft(head=head)
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
          answer = text or self.session.final_message
          if self.summary_asked:  # the reply to the one request for a clean final summary
            if text and len(text) >= 0.3 * len(self.best_summary):
              return text
            self.session.final_message = self.best_summary or answer
            return self.session.final_message
          if not self.is_chatter(text) and len(text) >= len(self.best_summary) * 0.6:
            self.best_summary = text
          try:
            follow = await self.after_answer(answer)
          except taskbridge.Cancelled:
            raise
          except Exception as exc:  # pylint: disable=broad-except
            await self.session.log(f"after_answer failed: {type(exc).__name__}: {str(exc)[:300]}")
            follow = None
          if follow is None:
            left = self.session.time_left()
            if (self.followups and self.is_chatter(text)
                and (left is None or left > self.finish_margin + 60)):
              # the last reply is a note about fixes: ask once for the summary the grader should read
              self.summary_asked = True
              follow = FINAL_SUMMARY_PROMPT
            else:
              return answer
          self.followups += 1
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
      self.asked_missing.update(missing)
      await self.session.log(f"gate: missing {missing}")
      return ("These paths, which the task appears to name as outputs, do not exist: " + ", ".join(missing)
              + ". If one is a deliverable of this task, create it exactly there. If one is not (for example the"
                " harness resolved a relative path against the wrong folder, or it is an input or an archive"
                " member), do NOT create it; you will not be asked again." + AGAIN)
    left = self.session.time_left()
    if self.format_checks < 3 and (left is None or left > self.finish_margin + 180):
      try:
        bad = await self.invalid_outputs()
      except taskbridge.Cancelled:
        raise
      except Exception as exc:  # pylint: disable=broad-except
        await self.session.log(f"format gate failed in the harness: {type(exc).__name__}: {exc}"[:300])
        bad = []
      if bad:
        self.format_checks += 1
        await self.session.log(f"format gate: {bad}"[:1500])
        return ("A strict reader of your output files (the kind graders use) reported the problems below. A"
                " deliverable that fails to open or validate usually scores zero, so fix each real one,"
                " preferably by generating the file with the format's standard library (e.g. ifcopenshell.api /"
                " ifcopenshell.guid.new() for IFC, openpyxl, python-docx, ezdxf) instead of hand-writing it, and"
                " re-open it to verify. If a finding concerns a scratch file, or is wrong for the format the"
                " task requires, leave that file unchanged and say so." + AGAIN + "\n\n" + "\n".join(bad))
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
            " every valid one in the files themselves (the inputs win over both of you) and re-run the relevant"
            " validation on the finished files. Leave unchanged anything a finding does not prove wrong, and do"
            " not restructure a deliverable to satisfy a preference." + AGAIN + "\n\n" + defects)


class Auditor(EngineerAgent):
  """A fresh-context checker that shares the task session but must not change files."""

  def adopt(self, parent: EngineerAgent) -> None:
    """Takes over what the parent learned about the endpoint and the workspace."""
    for name in ("output_ceiling", "context_window", "max_output", "token_limit", "effort", "responses",
                 "finish_margin", "mcp", "inputs", "started"):
      if hasattr(parent, name):
        setattr(self, name, getattr(parent, name))
    self.preview_map = getattr(parent, "preview_map", {})
    self.aliases = list(parent.aliases)
    # The task's own tools: only those whose name says they read are kept; anything that looks like it changes
    # state (or cannot be told) is neither offered nor callable.
    keep = {}
    for name, (server, t) in parent.mcp.items():
      tname = str(t.get("name", ""))
      if READ_TOOL_RE.search(tname) and not WRITE_TOOL_RE.search(tname):
        keep[name] = (server, t)
    self.blocked_tools = frozenset(set(parent.mcp) - set(keep))
    self.mcp = keep
    self.tools = [t for t in parent.tools
                  if (t.get("function") or {}).get("name") not in ("write_file", "edit_file")
                  and (t.get("function") or {}).get("name") not in self.blocked_tools]
    self.read_only = True
    self.used = 0
    self.deadline = time.time() + 1800

  async def compact(self, force: bool = False) -> bool:
    """Compacts with notes worded for a checker (the reference prompt asks for an engineer's work notes)."""
    old = serve.SUMMARY_PROMPT
    serve.SUMMARY_PROMPT = AUDIT_NOTES_PROMPT
    try:
      return await super().compact(force)
    finally:
      serve.SUMMARY_PROMPT = old

  async def bash(self, command: str, timeout_sec: int = 120) -> str:
    """bash, with the timeout cut to what the check has left."""
    cap = max(20, int(self.deadline - time.time()))
    return await super().bash(command, taskbridge.positive_int(timeout_sec, 120, cap))

  async def ask(self, seconds: float, **kwargs: Any) -> serve.Reply:
    """One model call that is cut off (CutOff) after about `seconds`, whatever the task has left."""
    seconds = max(20.0, seconds)
    left = self.session.time_left()
    if left is None:
      try:
        return await asyncio.wait_for(self.complete_soft(**kwargs), seconds)
      except asyncio.TimeoutError:
        raise CutOff() from None
    # complete() stops streaming `reserve` seconds before the task deadline
    return await self.complete_soft(reserve=max(self.finish_margin, left - seconds), **kwargs)

  async def audit(self, brief: str, minutes: float, steps: int) -> str | None:
    deadline = self.deadline = time.time() + minutes * 60
    self.messages = [
        {"role": "system", "content": AUDIT_SYSTEM.format(
            workdir=self.session.workdir, minutes=int(minutes), steps=steps,
            mcp=", and the task's read-only tools" if self.mcp else "")},
        {"role": "user", "content": brief},
    ]
    nudged = False
    cut = False  # a model call ran into the check's deadline
    while True:
      left = self.session.time_left()
      over = (cut or self.used >= steps or time.time() > deadline
              or (left is not None and left < self.finish_margin + 600))
      try:
        if over:
          self.messages.append({"role": "user", "content": (
              "Stop investigating now. Answer with your verified findings (or exactly NO_ISSUES).")})
          # no head: the checker's text must never become the session's final message
          reply = await self.ask(180, tool_choice="none")
        else:
          reply = await self.ask(deadline - time.time())
      except CutOff:
        if over:
          return None
        cut = True
        continue
      except (OutOfTime, openai.OpenAIError):
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
