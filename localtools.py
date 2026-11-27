"""Document tools that run inside the harness container.

Task containers ship whatever their authors installed, and often lack PDF,
spreadsheet or CAD libraries. These helpers pull a file out of the task
container over the protocol and parse or render it here, where the harness
bundle carries pinned libraries. Nothing here touches the network.
"""

from __future__ import annotations

import base64
import io
import json
import posixpath
import zipfile
from typing import Any

MAX_TEXT = 60_000  # characters one inspect call returns
RENDER_DPI = 110  # default page render resolution
MAX_SIDE = 2000  # longest side of an image sent to the model, in pixels

TEXT_EXT = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".yaml", ".yml", ".xml", ".html",
    ".htm", ".py", ".toml", ".ini", ".cfg", ".log", ".tex", ".sh", ".js",
    ".ts", ".sql", ".svg", ".geojson", ".kml", ".rst",
}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}


def ext(path: str) -> str:
  return posixpath.splitext(path.lower())[1]


def clip(text: str, limit: int = MAX_TEXT) -> str:
  if len(text) <= limit:
    return text
  half = limit // 2
  return (
      text[:half]
      + f"\n[... {len(text) - limit} characters omitted; ask for specific"
      " pages, sheets or a narrower range ...]\n"
      + text[-half:]
  )


def parse_range(spec: Any, count: int) -> list[int]:
  """Turns '1-3,7' (1-based) or an int into 0-based indexes within count."""
  if spec in (None, "", "all"):
    return list(range(count))
  out: list[int] = []
  for part in str(spec).replace(" ", "").split(","):
    if not part:
      continue
    if "-" in part:
      a, b = part.split("-", 1)
      lo = int(a) if a else 1
      hi = int(b) if b else count
      out += list(range(lo - 1, min(hi, count)))
    else:
      n = int(part)
      if 1 <= n <= count:
        out.append(n - 1)
  return sorted(set(i for i in out if 0 <= i < count))


# ---------------------------------------------------------------- PDF


def pdf_info(data: bytes) -> str:
  import pypdfium2 as pdfium  # pylint: disable=import-outside-toplevel

  doc = pdfium.PdfDocument(data)
  sizes = []
  for i in range(len(doc)):
    w, h = doc[i].get_size()
    sizes.append(f"{w / 72:.1f}x{h / 72:.1f}in")
  uniq = sorted(set(sizes))
  return f"PDF, {len(doc)} pages, page sizes: {', '.join(uniq[:6])}"


def pdf_text(data: bytes, pages: Any = None, tables: bool = True) -> str:
  import pdfplumber  # pylint: disable=import-outside-toplevel

  out = []
  with pdfplumber.open(io.BytesIO(data)) as pdf:
    idx = parse_range(pages, len(pdf.pages))
    out.append(f"[PDF with {len(pdf.pages)} pages; showing {len(idx)}]")
    for i in idx:
      page = pdf.pages[i]
      try:
        text = page.extract_text(layout=False) or ""
      except Exception as exc:  # pylint: disable=broad-except
        text = f"(text extraction failed: {exc})"
      words = len(text.split())
      out.append(
          f"\n===== page {i + 1} ({page.width / 72:.1f}x"
          f"{page.height / 72:.1f}in, {words} words) ====="
      )
      out.append(text.strip() or "(no text layer: use view to see it)")
      if tables:
        try:
          for t_n, table in enumerate(page.extract_tables()[:8]):
            rows = [
                " | ".join("" if c is None else str(c).replace("\n", " ")
                           for c in row)
                for row in table[:80]
            ]
            out.append(f"--- table {t_n + 1} on page {i + 1} ---")
            out.append("\n".join(rows))
        except Exception:  # pylint: disable=broad-except
          pass
      if words < 15:
        out.append("(little text here: this page is probably a drawing; use"
                   " view to look at it)")
  return "\n".join(out)


def pdf_search(data: bytes, query: str) -> str:
  import pdfplumber  # pylint: disable=import-outside-toplevel

  q = query.lower()
  hits = []
  with pdfplumber.open(io.BytesIO(data)) as pdf:
    for i, page in enumerate(pdf.pages):
      text = page.extract_text() or ""
      for line in text.splitlines():
        if q in line.lower():
          hits.append(f"p{i + 1}: {line.strip()}")
  return "\n".join(hits[:300]) or f"no line contains {query!r}"


def render_pdf(
    data: bytes, page: int, dpi: int = RENDER_DPI, crop: Any = None
) -> bytes:
  """Renders a 1-based page to PNG; crop = [x0, y0, x1, y1] as fractions."""
  import pypdfium2 as pdfium  # pylint: disable=import-outside-toplevel

  doc = pdfium.PdfDocument(data)
  if not 1 <= page <= len(doc):
    raise ValueError(f"page {page} is out of range 1-{len(doc)}")
  pg = doc[page - 1]
  w, h = pg.get_size()
  box = norm_crop(crop)
  # pick a scale so the (cropped) region's long side is about MAX_SIDE at most
  frac_w, frac_h = box[2] - box[0], box[3] - box[1]
  scale = dpi / 72
  if box != (0.0, 0.0, 1.0, 1.0):
    # a crop is a zoom: render it near full size, up to 400 dpi
    scale = max(scale, min(400 / 72, 0.8 * MAX_SIDE / max(w * frac_w, h * frac_h)))
  longest = max(w * frac_w, h * frac_h) * scale
  if longest > MAX_SIDE:
    scale *= MAX_SIDE / longest
  if box != (0.0, 0.0, 1.0, 1.0):
    crop_px = (box[0] * w, (1 - box[3]) * h, (1 - box[2]) * w, box[1] * h)
    bitmap = pg.render(scale=scale, crop=crop_px)
  else:
    bitmap = pg.render(scale=scale)
  img = bitmap.to_pil()
  return png(img)


def norm_crop(crop: Any) -> tuple[float, float, float, float]:
  if not crop:
    return (0.0, 0.0, 1.0, 1.0)
  if isinstance(crop, str):
    crop = json.loads(crop) if crop.strip().startswith("[") else [
        float(v) for v in crop.split(",")
    ]
  x0, y0, x1, y1 = [min(1.0, max(0.0, float(v))) for v in crop]
  if x1 <= x0 or y1 <= y0:
    raise ValueError("crop needs x0 < x1 and y0 < y1, as fractions 0-1")
  return (x0, y0, x1, y1)


def png(img: Any) -> bytes:
  img = img.convert("RGB") if img.mode not in ("RGB", "L") else img
  if max(img.size) > MAX_SIDE:
    img.thumbnail((MAX_SIDE, MAX_SIDE))
  buf = io.BytesIO()
  img.save(buf, format="PNG", optimize=True)
  return buf.getvalue()


def render_image(data: bytes, crop: Any = None) -> bytes:
  from PIL import Image  # pylint: disable=import-outside-toplevel

  img = Image.open(io.BytesIO(data))
  img.load()
  box = norm_crop(crop)
  if box != (0.0, 0.0, 1.0, 1.0):
    w, h = img.size
    img = img.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w),
                    int(box[3] * h)))
  return png(img)


# ---------------------------------------------------------------- Office


def xlsx_text(data: bytes, sheet: Any = None, cells: str | None = None) -> str:
  import openpyxl  # pylint: disable=import-outside-toplevel

  wb_f = openpyxl.load_workbook(io.BytesIO(data), data_only=False)
  wb_v = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
  names = wb_f.sheetnames
  out = [f"[workbook sheets: {names}]"]
  chosen = names
  if sheet:
    chosen = [n for n in names if n.lower() == str(sheet).lower()] or [
        n for n in names if str(sheet).lower() in n.lower()
    ]
    if not chosen:
      return f"no sheet {sheet!r}; sheets are {names}"
  for name in chosen:
    ws_f, ws_v = wb_f[name], wb_v[name]
    out.append(
        f"\n===== sheet {name!r} ({ws_f.max_row} rows x {ws_f.max_column}"
        f" cols, used range {ws_f.dimensions}) ====="
    )
    rng = ws_f[cells] if cells else ws_f.iter_rows()
    if cells and not isinstance(rng, tuple):
      rng = ((rng,),)
    elif cells and rng and not isinstance(rng[0], tuple):
      rng = (rng,)
    for row in rng:
      parts = []
      for c in row:
        if c.value is None:
          continue
        v = ws_v[c.coordinate].value
        if isinstance(c.value, str) and c.value.startswith("="):
          parts.append(f"{c.coordinate}={c.value} -> {v!r}")
        else:
          parts.append(f"{c.coordinate}={c.value!r}")
      if parts:
        out.append("; ".join(parts))
    merged = list(ws_f.merged_cells.ranges)[:30]
    if merged:
      out.append(f"(merged: {', '.join(str(m) for m in merged)})")
  return "\n".join(out)


def docx_text(data: bytes) -> str:
  import docx  # pylint: disable=import-outside-toplevel
  from docx.oxml.ns import qn  # pylint: disable=import-outside-toplevel

  d = docx.Document(io.BytesIO(data))
  out = []
  body = d.element.body
  t_n = 0
  for child in body.iterchildren():
    if child.tag == qn("w:p"):
      p = docx.text.paragraph.Paragraph(child, d)
      text = p.text.strip()
      if text:
        style = p.style.name if p.style is not None else ""
        prefix = f"[{style}] " if style and style != "Normal" else ""
        out.append(prefix + text)
    elif child.tag == qn("w:tbl"):
      t_n += 1
      table = docx.table.Table(child, d)
      out.append(f"--- table {t_n} ---")
      for row in table.rows[:120]:
        out.append(" | ".join(c.text.strip().replace("\n", " ")
                              for c in row.cells))
  for s_n, section in enumerate(d.sections):
    for kind, part in (("header", section.header), ("footer", section.footer)):
      try:
        text = " ".join(p.text for p in part.paragraphs).strip()
      except Exception:  # pylint: disable=broad-except
        text = ""
      if text:
        out.append(f"[section {s_n + 1} {kind}] {text}")
  try:
    comments = zipfile.ZipFile(io.BytesIO(data)).read("word/comments.xml")
    out.append("[document has comments.xml: "
               f"{comments.count(b'<w:comment ')} comments]")
  except KeyError:
    pass
  return "\n".join(out) or "(empty document)"


def pptx_text(data: bytes) -> str:
  import pptx  # pylint: disable=import-outside-toplevel

  prs = pptx.Presentation(io.BytesIO(data))
  out = []
  for i, slide in enumerate(prs.slides):
    out.append(f"===== slide {i + 1} =====")
    for shape in slide.shapes:
      if shape.has_text_frame and shape.text_frame.text.strip():
        out.append(shape.text_frame.text.strip())
      if getattr(shape, "has_table", False) and shape.has_table:
        for row in shape.table.rows:
          out.append(" | ".join(c.text for c in row.cells))
  return "\n".join(out)


# ---------------------------------------------------------------- CAD / BIM


def dxf_text(data: bytes) -> str:
  import ezdxf  # pylint: disable=import-outside-toplevel
  from ezdxf import recover  # pylint: disable=import-outside-toplevel

  try:
    doc, _ = recover.read(io.BytesIO(data))
  except Exception:  # pylint: disable=broad-except
    doc = ezdxf.read(io.StringIO(data.decode("utf-8", "replace")))
  msp = doc.modelspace()
  out = [f"[DXF {doc.dxfversion}, units code {doc.header.get('$INSUNITS')}]"]
  counts: dict[str, int] = {}
  layers: dict[str, int] = {}
  texts = []
  dims = []
  inserts: dict[str, int] = {}
  for e in msp:
    t = e.dxftype()
    counts[t] = counts.get(t, 0) + 1
    layers[e.dxf.layer] = layers.get(e.dxf.layer, 0) + 1
    if t in ("TEXT", "MTEXT", "ATTRIB"):
      try:
        txt = e.plain_text() if t == "MTEXT" else e.dxf.text
        ins = e.dxf.insert
        texts.append(f"{txt!r} @({ins[0]:.2f},{ins[1]:.2f}) layer={e.dxf.layer}")
      except Exception:  # pylint: disable=broad-except
        pass
    elif t == "DIMENSION":
      try:
        dims.append(f"{e.get_measurement():.3f} layer={e.dxf.layer}")
      except Exception:  # pylint: disable=broad-except
        pass
    elif t == "INSERT":
      inserts[e.dxf.name] = inserts.get(e.dxf.name, 0) + 1
  try:
    from ezdxf import bbox  # pylint: disable=import-outside-toplevel

    ext_box = bbox.extents(msp)
    out.append(f"extents: {ext_box.extmin} to {ext_box.extmax}")
  except Exception:  # pylint: disable=broad-except
    pass
  out.append(f"entities: {counts}")
  out.append(f"layers (entity counts): {layers}")
  if inserts:
    out.append(f"block inserts: {inserts}")
  if dims:
    out.append(f"dimensions ({len(dims)}): " + "; ".join(dims[:200]))
  if texts:
    out.append(f"texts ({len(texts)}):\n" + "\n".join(texts[:600]))
  return "\n".join(out)


def ifc_text(data: bytes) -> str:
  try:
    import ifcopenshell  # pylint: disable=import-outside-toplevel
  except ImportError:
    return ifc_text_plain(data)
  try:
    f = ifcopenshell.file.from_string(data.decode("utf-8", "replace"))
  except Exception:  # pylint: disable=broad-except
    return ifc_text_plain(data)
  out = [f"[IFC schema {f.schema}]"]
  counts: dict[str, int] = {}
  for e in f:
    counts[e.is_a()] = counts.get(e.is_a(), 0) + 1
  top = sorted(counts.items(), key=lambda kv: -kv[1])
  out.append("entity counts: " + ", ".join(f"{k}={v}" for k, v in top[:60]))
  for kind in ("IfcProject", "IfcSite", "IfcBuilding", "IfcBuildingStorey"):
    for e in f.by_type(kind)[:20]:
      out.append(f"{kind} #{e.id()} name={e.Name!r}")
  try:
    units = f.by_type("IfcSIUnit")
    out.append("SI units: " + ", ".join(
        f"{u.UnitType}:{u.Prefix or ''}{u.Name}" for u in units[:12]))
  except Exception:  # pylint: disable=broad-except
    pass
  shown = 0
  for e in f.by_type("IfcProduct"):
    if e.is_a() in ("IfcSite", "IfcBuilding", "IfcBuildingStorey"):
      continue
    if shown >= 300:
      out.append("[... more products omitted ...]")
      break
    out.append(f"{e.is_a()} #{e.id()} name={e.Name!r} tag="
               f"{getattr(e, 'Tag', None)!r}")
    shown += 1
  return "\n".join(out)


def ifc_text_plain(data: bytes) -> str:
  text = data.decode("utf-8", "replace")
  counts: dict[str, int] = {}
  for line in text.splitlines():
    if line.startswith("#") and "=" in line:
      kind = line.split("=", 1)[1].strip().split("(", 1)[0]
      counts[kind] = counts.get(kind, 0) + 1
  top = sorted(counts.items(), key=lambda kv: -kv[1])
  return "[IFC (parsed as text)] entity counts: " + ", ".join(
      f"{k}={v}" for k, v in top[:80])


# ---------------------------------------------------------------- validation


def _ifc_problems(data: bytes) -> list[str]:
  try:
    import ifcopenshell  # pylint: disable=import-outside-toplevel
    import ifcopenshell.validate  # pylint: disable=import-outside-toplevel
  except ImportError:
    return []
  try:
    f = ifcopenshell.file.from_string(data.decode("utf-8", "replace"))
  except Exception as exc:  # pylint: disable=broad-except
    return [f"does not parse as IFC: {str(exc)[:300]}"]
  out: list[str] = []
  try:
    log = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(f, log, express_rules=True)
    kinds: dict[str, list[str]] = {}
    for st in log.statements:
      msg = str(st.get("message", st) if isinstance(st, dict) else st)
      lines = [ln.strip() for ln in msg.splitlines() if ln.strip()]
      attr = next((ln.split(":", 1)[0].replace("<attribute ", "attribute ") for ln in lines
                   if ln.startswith("<attribute")), "")
      key = next((ln for ln in lines if not ln.startswith(("On instance", "#", "^", "With attribute", "<"))),
                 lines[0] if lines else "error")[:160]
      if attr:
        key = f"{attr}: {key}"
      inst = next((ln for ln in lines if ln.startswith("#")), "")[:160]
      kinds.setdefault(key, []).append(inst)
    for key, insts in list(kinds.items())[:12]:
      out.append(f"schema: {key} ({len(insts)}x, e.g. {insts[0]})")
  except Exception as exc:  # pylint: disable=broad-except
    out.append(f"schema validator crashed: {str(exc)[:200]}")
  try:
    if not f.by_type("IfcProject"):
      out.append("no IfcProject")
    noplace = [f"{e.is_a()} {e.Name!r}" for e in f.by_type("IfcProduct")
               if getattr(e, "ObjectPlacement", None) is None]
    if noplace:
      out.append(f"{len(noplace)} products have no ObjectPlacement (readers crash or mislocate them): "
                 + ", ".join(noplace[:8]))
    if f.by_type("IfcProduct") and not f.by_type("IfcRelContainedInSpatialStructure") \
        and not f.by_type("IfcRelAggregates"):
      out.append("products are not linked into the spatial tree")
  except Exception as exc:  # pylint: disable=broad-except
    out.append(f"model walk failed: {str(exc)[:200]}")
  return out


def validate(path: str, data: bytes) -> list[str]:
  """Problems a strict reader of this file format would hit ([] = looks valid)."""
  e = ext(path)
  try:
    if not data:
      return ["file is empty"]
    if e == ".ifc":
      return _ifc_problems(data)
    if e == ".json":
      json.loads(data.decode("utf-8"))
    elif e in (".csv",):
      import csv  # pylint: disable=import-outside-toplevel
      rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
      widths = {len(r) for r in rows if r}
      if len(widths) > 1:
        return [f"rows have differing column counts {sorted(widths)[:6]}"]
    elif e in (".xlsx", ".xlsm"):
      import openpyxl  # pylint: disable=import-outside-toplevel
      wb = openpyxl.load_workbook(io.BytesIO(data))
      bad = []
      for ws in wb.worksheets:
        for row in ws.iter_rows():
          for c in row:
            v = c.value
            if isinstance(v, str) and v.startswith("#") and v.rstrip("!?0/").upper() in (
                "#REF", "#VALUE", "#DIV", "#NAME", "#N/A", "#NUM", "#NULL"):
              bad.append(f"{ws.title}!{c.coordinate}={v}")
      if bad:
        return [f"error values in cells: {', '.join(bad[:10])}"]
    elif e == ".docx":
      import docx  # pylint: disable=import-outside-toplevel
      docx.Document(io.BytesIO(data))
    elif e == ".pptx":
      import pptx  # pylint: disable=import-outside-toplevel
      pptx.Presentation(io.BytesIO(data))
    elif e == ".pdf":
      import pdfplumber  # pylint: disable=import-outside-toplevel
      with pdfplumber.open(io.BytesIO(data)) as pdf:
        if not pdf.pages:
          return ["PDF has no pages"]
    elif e == ".dxf":
      from ezdxf import recover  # pylint: disable=import-outside-toplevel
      doc, auditor = recover.read(io.BytesIO(data))
      errs = [str(x.message)[:120] for x in auditor.errors][:6]
      if errs:
        return ["DXF audit errors: " + "; ".join(errs)]
      del doc
    elif e == ".zip":
      z = zipfile.ZipFile(io.BytesIO(data))
      badf = z.testzip()
      if badf:
        return [f"corrupt member {badf}"]
  except Exception as exc:  # pylint: disable=broad-except
    return [f"cannot be opened as {e or 'this format'}: {type(exc).__name__}: {str(exc)[:240]}"]
  return []


# ---------------------------------------------------------------- dispatch


def zip_text(data: bytes) -> str:
  z = zipfile.ZipFile(io.BytesIO(data))
  rows = [f"{i.file_size:>10}  {i.filename}" for i in z.infolist()[:400]]
  return f"[zip archive, {len(z.infolist())} entries]\n" + "\n".join(rows)


def inspect(path: str, data: bytes, pages: Any = None, sheet: Any = None,
            cells: str | None = None, query: str | None = None) -> str:
  e = ext(path)
  if e == ".pdf":
    if query:
      return pdf_search(data, query)
    head = pdf_info(data)
    return head + "\n" + clip(pdf_text(data, pages))
  if e in (".xlsx", ".xlsm"):
    return clip(xlsx_text(data, sheet, cells))
  if e == ".docx":
    return clip(grep(docx_text(data), query))
  if e == ".pptx":
    return clip(grep(pptx_text(data), query))
  if e == ".dxf":
    return clip(grep(dxf_text(data), query))
  if e in (".ifc", ".ifczip"):
    return clip(grep(ifc_text(data), query))
  if e == ".zip":
    return clip(zip_text(data))
  if e in IMAGE_EXT:
    from PIL import Image  # pylint: disable=import-outside-toplevel

    img = Image.open(io.BytesIO(data))
    return (f"image {img.size[0]}x{img.size[1]} {img.mode}; call view to look"
            " at it")
  if e == ".xls":
    try:
      import xlrd  # pylint: disable=import-outside-toplevel

      book = xlrd.open_workbook(file_contents=data)
      out = []
      for sh in book.sheets():
        out.append(f"===== sheet {sh.name!r} =====")
        for r in range(sh.nrows):
          out.append(" | ".join(str(v) for v in sh.row_values(r)))
      return clip("\n".join(out))
    except Exception as exc:  # pylint: disable=broad-except
      return f"could not read .xls: {exc}"
  text = data.decode("utf-8", "replace")
  return clip(grep(text, query))


def grep(text: str, query: str | None) -> str:
  if not query:
    return text
  q = query.lower()
  lines = text.splitlines()
  hits = [f"{n + 1}: {line}" for n, line in enumerate(lines) if q in line.lower()]
  return "\n".join(hits[:400]) or f"no line contains {query!r}"


def render(path: str, data: bytes, page: int | None = None,
           crop: Any = None, dpi: int | None = None) -> bytes:
  e = ext(path)
  if e == ".pdf":
    return render_pdf(data, int(page or 1), int(dpi or RENDER_DPI), crop)
  if e in IMAGE_EXT:
    return render_image(data, crop)
  if e == ".dxf":
    return render_dxf(data, crop)
  raise ValueError(f"view supports PDF pages, images and DXF, not {e or path}")


def render_dxf(data: bytes, crop: Any = None) -> bytes:
  import matplotlib  # pylint: disable=import-outside-toplevel

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
  from ezdxf import recover  # pylint: disable=import-outside-toplevel
  from ezdxf.addons.drawing import RenderContext, Frontend  # pylint: disable=import-outside-toplevel
  from ezdxf.addons.drawing.matplotlib import MatplotlibBackend  # pylint: disable=import-outside-toplevel
  from PIL import Image  # pylint: disable=import-outside-toplevel

  doc, _ = recover.read(io.BytesIO(data))
  fig = plt.figure(figsize=(14, 10), dpi=140)
  ax = fig.add_axes([0, 0, 1, 1])
  Frontend(RenderContext(doc), MatplotlibBackend(ax)).draw_layout(
      doc.modelspace(), finalize=True)
  buf = io.BytesIO()
  fig.savefig(buf, format="png", facecolor="white")
  plt.close(fig)
  return render_image(buf.getvalue(), crop) if crop else png(
      Image.open(io.BytesIO(buf.getvalue())))


def data_url(png_bytes: bytes) -> str:
  return "data:image/png;base64," + base64.b64encode(png_bytes).decode()
