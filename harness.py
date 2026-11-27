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
4. An independent review: a fresh-context reviewer reads the instruction and
   the delivered files and lists concrete defects; the agent then fixes them.
   This repeats while time and the review budget allow.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import sys
from typing import Any

import openai
import serve
from serve import Agent, CutOff, OutOfTime, StreamCut, parse_arguments, tool, visible
import taskbridge
import localtools

REVIEW_ROUNDS = int(os.environ.get("REVIEW_ROUNDS", "2"))
# a review starts only if this many minutes remain
REVIEW_MIN_LEFT_MIN = float(os.environ.get("REVIEW_MIN_LEFT_MIN", "20"))
KEEP_IMAGES = int(os.environ.get("KEEP_IMAGES", "6"))
IMAGE_TOKENS = 1600  # rough prompt cost of one image
REVIEW_CHARS = int(os.environ.get("REVIEW_CHARS", "120000"))
SURVEY_FILES = 400

EXTRA_SYSTEM = """

Engineering practice (you are graded like a licensed engineer's deliverable):
- First make a checklist of every deliverable, required file path, format, section, and number the task asks for, \
and of every input file that bears on it. Satisfy each item explicitly.
- Read the governing documents (specifications, code excerpts, approval letters, schedules) before you compute. Use \
the project's own criteria, values and clause numbers over general rules of thumb, and cite them.
- Drawings: use the inspect tool for the text layer, and the view tool to look at sheets, zooming (crop) into title \
blocks, callouts, schedules and dimensions. Never guess a value that a drawing shows.
- Compute with code (python3 in the container), keep units explicit and convert carefully; sanity-check magnitudes.
- State assumptions, design criteria and results clearly in the deliverable; give numbers with units.
- When asked for a decision, a verdict or a pass/fail, give it plainly, with the governing check.
- Never stop early: you have hours. Use the time to verify, cross-check against the inputs, and polish.

Harness tools (they run outside the container, on a copy of the file):
- inspect: text of a PDF (per page, with tables; or search with query), DOCX, PPTX, XLSX (formulas and values), DXF \
(layers, texts, dimensions), IFC (entities), ZIP listings.
- view: look at a PDF page, an image or a DXF drawing; crop=[x0,y0,x1,y1] (fractions of the page, origin top-left) \
zooms in."""

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

REVIEW_SYSTEM = """You are a senior licensed professional engineer doing an independent QA review of a junior \
engineer's deliverables before they are issued. You are given the task exactly as the client wrote it, the junior's \
summary, and the contents of the files they delivered. You cannot see the input files, so judge against the task \
text and internal consistency.

List only concrete, fixable defects that would cost credit with a strict grader, most important first:
- a required file, section, table, field, or answer that is missing, misplaced, misnamed, or in the wrong format;
- a requirement in the task that is not explicitly addressed;
- numbers that are inconsistent, implausible, have wrong or missing units, or conflict between files;
- a decision or conclusion that is hedged, missing, or not supported by a stated check;
- placeholders, TODOs, unreadable or empty files.
Be specific (file, location, what is wrong, what it should be). Do not ask for cosmetic changes. If the deliverables \
fully meet the task, answer exactly NO_ISSUES."""

PATH_RE = re.compile(r"(?<![\w.])(/(?:app|filesystem|workspace|output|data|home|mnt|tmp)/[\w./@+-]*[\w/])")


def wanted_paths(instruction: str) -> list[str]:
  """Absolute paths the instruction names (likely deliverables or inputs)."""
  seen: list[str] = []
  for m in PATH_RE.finditer(instruction):
    p = m.group(1).rstrip(".,;:)")
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
    self.cache: dict[str, bytes] = {}

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
    named = wanted_paths(self.session.instruction)
    named_note = ""
    if named:
      named_note = "\nPaths named in the task: " + ", ".join(named)
    return (
        f"Project files ({len(lines)} shown):\n" + "\n".join(lines)
        + f"\n\nTask container tooling: {tooling.strip()}"
        + "\n(Office/PDF libraries missing in the container? Use inspect/view to read files, and write"
          " deliverables with whatever is available, e.g. python-docx or openpyxl if present, otherwise build"
          " the Office XML zip with python's zipfile.)"
        + named_note
    )

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
    named = wanted_paths(self.session.instruction)
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
    named = set(wanted_paths(self.session.instruction))
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
      parts.append(f"\n===== {path} =====\n{text}")
    return "\n".join(parts)

  async def review(self, summary: str) -> str | None:
    """Fresh-context QA review; returns defects, or None when clean/failed."""
    try:
      dump = await self.deliverable_dump()
    except taskbridge.TaskError as exc:
      await self.session.log(f"review: dump failed: {exc}")
      return None
    messages = [
        {"role": "system", "content": REVIEW_SYSTEM},
        {"role": "user", "content": (
            "THE TASK:\n" + self.session.instruction
            + "\n\nTHE ENGINEER'S SUMMARY:\n" + (summary or "(none)")
            + "\n\nTHE DELIVERED FILES:\n" + dump)},
    ]
    left = self.session.time_left()
    reserve = self.finish_margin + (10 * 60 if left else 0)
    try:
      reply = await self.stream(messages, "none", reserve)
    except (openai.APIError, StreamCut, asyncio.TimeoutError, CutOff) as exc:
      await self.session.log(f"review failed: {str(exc)[:200]}")
      return None
    await self.report(reply)
    text = visible(reply.text).strip()
    await self.session.log(f"review {self.reviews}: {text[:600]}")
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
    if self.reviews >= REVIEW_ROUNDS or self.minutes_left() < REVIEW_MIN_LEFT_MIN:
      return None
    self.reviews += 1
    defects = await self.review(summary)
    if not defects:
      return None
    return ("An independent senior engineer reviewed your deliverables (without seeing the input files) and"
            " reported the issues below. Check each against the task and the inputs; fix every valid one in"
            " the files themselves, ignore any that are wrong (the inputs win), re-open the changed files to"
            " verify, then answer with your updated final summary.\n\n" + defects)


async def solve(session: taskbridge.Session) -> str:
  print(f"task: workdir {session.workdir}, model {serve.MODEL},"
        f" {len(session.mcp_servers)} MCP server(s)", file=sys.stderr)
  return await EngineerAgent(session).run()


if __name__ == "__main__":
  taskbridge.run(solve)
