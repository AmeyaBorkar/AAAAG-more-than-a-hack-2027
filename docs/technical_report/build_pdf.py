"""Typeset AAAAG.pdf, the short technical document, from AAAAG_en.md and AAAAG_es.md.

    python docs/technical_report/build_pdf.py            # writes docs/technical_report/AAAAG.pdf
    python docs/technical_report/build_pdf.py --check    # also prints the style-guide checks

English comes first, then the full Spanish translation on a new page. Page 1 links to the Spanish
part and the Spanish title links back. The layout follows the team's style guide: A4, Times New
Roman, 3.5 cm side margins, body 12/17 pt, headings 15 pt bold, tables with three horizontal rules,
a boxed Summary, and a footer "Page X of Y". Everything is black except links: every link (the
language cross-links, the repository URL, dataset sources) is a real link annotation whose text is
Office hyperlink blue (`LINK_HEX`, #0563C1), with no underline and no border.

The repository URL is the token REPO_URL in both texts. Set the constant `REPO_URL` below once and
every occurrence becomes the URL, as a link when it starts with http.

Markdown subset (both files):
- front matter between `---` lines, `key: value` per line: lang, anchor, outline, header, footer
  (with {page} and {total}), title, meta (repeatable), crosslink (`anchor | text`);
- `## text` is a section heading; any other line run is a paragraph (blank line ends it);
- inline `**bold**`, `` `code` `` and `[text](url)` links; `{page:en}` and `{page:es}` become page
  numbers;
- `- item` and `1. item` lists (continuation lines are indented);
- `Table: caption`, then a pipe table whose second row sets alignment (`---:` is right-aligned),
  then an optional `Note: text` line;
- fenced code blocks (three backticks): commands, Courier New 10.5 pt with a left rule;
- `::: summary` ... `:::`, the boxed Summary;
- `::: figure` ... `:::`, the pipeline schematic: one line per box, its text lines separated
  by `|`, then `group: 1-2 | label` lines and one `caption: text` line.

Fonts: Times New Roman and Courier New from C:/Windows/Fonts, or from the folder in the
environment variable AAAAG_FONT_DIR (needed on other systems). Fonts are embedded as subsets.

Determinism: reportlab's invariant mode fixes the dates and document id, so the same inputs give
the same bytes. The build runs until the page numbers it prints stop changing (two passes).
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from reportlab import rl_config
from reportlab.lib.colors import black
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)

REPO_URL = "https://github.com/AmeyaBorkar/AAAAG-more-than-a-hack-2027"  # the only place to set it
LINK_HEX = "#0563C1"  # Office's hyperlink blue: link text only, everything else is black

HERE = Path(__file__).resolve().parent
SOURCES = (HERE / "AAAAG_en.md", HERE / "AAAAG_es.md")
OUTPUT = HERE / "AAAAG.pdf"

PDF_TITLE = "AAAAG: frame-level ball possession from automatic tracking"
PDF_AUTHOR = "Team AAAAG"
PDF_SUBJECT = (
    "Technical document for More than a Hack 2027, Phase 1, in English and Spanish "
    "(Documento técnico, en inglés y en español)"
)

# Geometry and type (style guide, part D).
PAGE_W, PAGE_H = A4
MARGIN_TB = 2.5 * cm
MARGIN_LR = 3.5 * cm
MEASURE = PAGE_W - 2 * MARGIN_LR
HEADER_BASELINE = PAGE_H - 1.55 * cm
FOOTER_BASELINE = 1.45 * cm
BODY, BODY_LEADING = 12, 17
HEADING, HEADING_LEADING = 15, 18
TITLE, TITLE_LEADING = 18, 22
SMALL, SMALL_LEADING = 11, 14
TABLE_LEADING = 13.5
CODE, CODE_LEADING = 10.5, 13
RUNNING = 10
RULE_THIN, RULE_TABLE, RULE_BOX = 0.5, 1.0, 0.75
FIGURE_TEXT, FIGURE_LEADING = 10, 12

FONT_FILES = {
    "TNR": "times.ttf",
    "TNR-Bold": "timesbd.ttf",
    "TNR-Italic": "timesi.ttf",
    "TNR-BoldItalic": "timesbi.ttf",
    "CourierNew": "cour.ttf",
}


def font_dir() -> Path:
    """The folder that holds the Times New Roman and Courier New TrueType files."""
    env = os.environ.get("AAAAG_FONT_DIR")
    if env:
        folder = Path(env)
    elif sys.platform == "win32":
        folder = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    else:
        raise SystemExit(
            "build_pdf: set AAAAG_FONT_DIR to a folder containing "
            + ", ".join(sorted(FONT_FILES.values()))
            + " (Times New Roman and Courier New)."
        )
    missing = [name for name in FONT_FILES.values() if not (folder / name).is_file()]
    if missing:
        raise SystemExit(
            f"build_pdf: {folder} lacks {', '.join(missing)}. Set AAAAG_FONT_DIR to a folder with "
            "the Times New Roman and Courier New TrueType files."
        )
    return folder


def register_fonts() -> None:
    folder = font_dir()
    for name, file in FONT_FILES.items():
        pdfmetrics.registerFont(TTFont(name, str(folder / file)))
    pdfmetrics.registerFontFamily(
        "TNR", normal="TNR", bold="TNR-Bold", italic="TNR-Italic", boldItalic="TNR-BoldItalic"
    )
    pdfmetrics.registerFontFamily(
        "CourierNew", normal="CourierNew", bold="CourierNew", italic="CourierNew"
    )


# ---------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------


@dataclass
class Block:
    kind: str  # heading, para, bullets, numbers, table, code, summary, figure
    text: str = ""
    items: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    align: list[str] = field(default_factory=list)
    caption: str = ""
    note: str = ""
    groups: list[tuple[int, int, str]] = field(default_factory=list)
    children: list[Block] = field(default_factory=list)


@dataclass
class Document:
    meta: dict[str, list[str]]
    blocks: list[Block]

    def one(self, key: str) -> str:
        return self.meta[key][0]


LIST_ITEM = re.compile(r"^(- |\d+\. )")


def parse(text: str) -> Document:
    lines = text.splitlines()
    meta: dict[str, list[str]] = {}
    start = 0
    if lines and lines[0].strip() == "---":
        end = lines.index("---", 1)
        for line in lines[1:end]:
            key, _, value = line.partition(":")
            meta.setdefault(key.strip(), []).append(value.strip())
        start = end + 1
    return Document(meta, parse_blocks(lines[start:]))


def parse_blocks(lines: list[str]) -> list[Block]:
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
        elif s.startswith(":::"):
            end = next(j for j in range(i + 1, len(lines)) if lines[j].strip() == ":::")
            inner = lines[i + 1 : end]
            if s[3:].strip() == "summary":
                blocks.append(Block("summary", children=parse_blocks(inner)))
            else:
                blocks.append(parse_figure(inner))
            i = end + 1
        elif s.startswith("```"):
            end = next(j for j in range(i + 1, len(lines)) if lines[j].strip().startswith("```"))
            blocks.append(Block("code", items=[line.rstrip() for line in lines[i + 1 : end]]))
            i = end + 1
        elif s.startswith("## "):
            blocks.append(Block("heading", text=s[3:].strip()))
            i += 1
        elif s.startswith("Table: "):
            block = Block("table", caption=s[len("Table: ") :])
            i += 1
            raw = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                raw.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            block.rows = [raw[0]] + raw[2:]
            block.align = ["right" if c.endswith(":") else "left" for c in raw[1]]
            if i < len(lines) and lines[i].strip().startswith("Note: "):
                note = [lines[i].strip()[len("Note: ") :]]
                i += 1
                while i < len(lines) and lines[i].strip():
                    note.append(lines[i].strip())
                    i += 1
                block.note = " ".join(note)
            blocks.append(block)
        elif LIST_ITEM.match(s):
            kind = "bullets" if s.startswith("- ") else "numbers"
            items: list[str] = []
            while i < len(lines) and lines[i].strip():
                line = lines[i]
                if LIST_ITEM.match(line.strip()) and not line.startswith(" "):
                    items.append(LIST_ITEM.sub("", line.strip(), count=1))
                else:
                    items[-1] += " " + line.strip()
                i += 1
            blocks.append(Block(kind, items=items))
        else:
            para = []
            while i < len(lines) and lines[i].strip():
                para.append(lines[i].strip())
                i += 1
            blocks.append(Block("para", text=" ".join(para)))
    return blocks


def parse_figure(lines: list[str]) -> Block:
    block = Block("figure")
    for line in (raw.strip() for raw in lines):
        if not line:
            continue
        if line.startswith("caption:"):
            block.caption = line[len("caption:") :].strip()
        elif line.startswith("group:"):
            span, _, label = line[len("group:") :].partition("|")
            first, _, last = span.strip().partition("-")
            block.groups.append((int(first), int(last or first), label.strip()))
        else:
            block.items.append(line)
    return block


# ---------------------------------------------------------------------------------------------
# Inline markup
# ---------------------------------------------------------------------------------------------

BOLD = re.compile(r"\*\*(.+?)\*\*")
CODE_SPAN = re.compile(r"`([^`]+)`")
LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def link(target: str, text: str) -> str:
    """Paragraph markup for a link: blue text, no underline, a real link annotation."""
    return f'<a href="{target}" color="{LINK_HEX}" underline="0">{text}</a>'


def substitute(text: str, pages: dict[str, int]) -> str:
    for key, page in pages.items():
        text = text.replace("{page:" + key + "}", str(page))
    return text


# A number stays on one line with its unit, and Spanish digit groups ("145 000") stay together.
NBSP = "\u00a0"
UNBREAKABLE = re.compile(r"(?<=\d) (?=%|s\b|m\b|m/s\b|Hz\b|\d{3}\b)")


# Names that must not break across lines.
KEEP_TOGETHER = ("PFF FC", "CC BY 4.0", "P(H, A, D)")
# A formula such as "½ W + ⅙ (G + G′ + G″)" or "3 × 2" never breaks across lines.
FORMULA_SPACE = re.compile(r"(?<=[½⅙]) | (?=[+×=] )|(?<= [+×=]) ")


def markup(text: str) -> str:
    """Markdown inline subset to reportlab paragraph markup."""
    text = FORMULA_SPACE.sub(NBSP, UNBREAKABLE.sub(NBSP, text))
    for name in KEEP_TOGETHER:
        text = text.replace(name, name.replace(" ", NBSP))
    out = html.escape(text, quote=False)
    out = CODE_SPAN.sub(r'<font face="CourierNew" size="11">\1</font>', out)
    out = BOLD.sub(r"<b>\1</b>", out)
    out = LINK.sub(lambda m: link(m.group(2), m.group(1)), out)
    if REPO_URL.startswith("http"):
        url = html.escape(REPO_URL)
        return out.replace("REPO_URL", link(url, url))
    return out.replace("REPO_URL", html.escape(REPO_URL))


def plain(text: str) -> str:
    text = LINK.sub(r"\1", text)
    return CODE_SPAN.sub(r"\1", BOLD.sub(r"\1", text)).replace("REPO_URL", REPO_URL)


# ---------------------------------------------------------------------------------------------
# Styles and flowables
# ---------------------------------------------------------------------------------------------


def styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle(
        "body",
        fontName="TNR",
        fontSize=BODY,
        leading=BODY_LEADING,
        spaceAfter=6,
        alignment=TA_LEFT,
        textColor=black,
        allowWidows=0,
        allowOrphans=0,
    )
    s = {"body": base}
    s["heading"] = ParagraphStyle(
        "heading",
        parent=base,
        fontName="TNR-Bold",
        fontSize=HEADING,
        leading=HEADING_LEADING,
        spaceBefore=18,
        spaceAfter=6,
        keepWithNext=1,
    )
    s["title"] = ParagraphStyle(
        "title", parent=base, fontName="TNR-Bold", fontSize=TITLE, leading=TITLE_LEADING
    )
    s["meta"] = ParagraphStyle(
        "meta", parent=base, fontSize=SMALL, leading=SMALL_LEADING, spaceAfter=0
    )
    s["item"] = ParagraphStyle(
        "item", parent=base, leftIndent=18, bulletIndent=0, bulletFontName="TNR", spaceAfter=2
    )
    s["caption"] = ParagraphStyle(
        "caption",
        parent=base,
        fontSize=SMALL,
        leading=SMALL_LEADING,
        spaceBefore=4,
        spaceAfter=4,
        keepWithNext=1,
    )
    s["note"] = ParagraphStyle(
        "note", parent=base, fontSize=SMALL, leading=SMALL_LEADING, spaceBefore=4, spaceAfter=0
    )
    s["cell"] = ParagraphStyle(
        "cell", parent=base, fontSize=SMALL, leading=TABLE_LEADING, spaceAfter=0
    )
    s["cell_right"] = ParagraphStyle("cell_right", parent=s["cell"], alignment=TA_RIGHT)
    s["code"] = ParagraphStyle(
        "code", parent=base, fontName="CourierNew", fontSize=CODE, leading=CODE_LEADING
    )
    s["figure"] = ParagraphStyle(
        "figure",
        parent=base,
        fontSize=FIGURE_TEXT,
        leading=FIGURE_LEADING,
        alignment=TA_CENTER,
        spaceAfter=0,
    )
    return s


class Group(Flowable):
    """Flowables that move as one block and never split: a table with its caption and note, or
    the figure with its caption. A floating group that does not fit lets the paragraphs after
    it, up to the next heading, go first (see Builder.handle_flowable), as LaTeX floats do."""

    def __init__(self, parts: list, space_after: float = 10, floating: bool = True):
        super().__init__()
        self.parts, self.space_after, self.floating = parts, space_after, floating
        self.layout: list[tuple[float, float]] = []

    def wrap(self, avail_w, avail_h):
        self.layout, height = [], 0.0
        for k, part in enumerate(self.parts):
            _, h = part.wrap(avail_w, avail_h)
            before = part.getSpaceBefore() if k else 0.0
            after = part.getSpaceAfter() if k < len(self.parts) - 1 else 0.0
            self.layout.append((before, h))
            height += before + h + after
        self.width, self.height = avail_w, height
        return avail_w, height

    def split(self, avail_w, avail_h):
        return []

    def getSpaceBefore(self):
        return self.parts[0].getSpaceBefore()

    def getSpaceAfter(self):
        return self.space_after

    def draw(self) -> None:
        y = self.height
        for k, (part, (before, h)) in enumerate(zip(self.parts, self.layout, strict=True)):
            y -= before + h
            part.drawOn(self.canv, 0, y)
            if k < len(self.parts) - 1:
                y -= part.getSpaceAfter()


def body(text: str, style: ParagraphStyle) -> Paragraph:
    """A body paragraph; a floating group may let it go first."""
    para = Paragraph(text, style)
    para.pullable = True
    return para


def outlined(text: str, style: ParagraphStyle, key: str, title: str, level: int) -> Paragraph:
    """A paragraph that is a link target (`key`) and an entry of the PDF outline."""
    para = Paragraph(f'<a name="{key}"/>' + text, style)
    para.outline = (key, title, level)
    return para


class PipelineFigure(Flowable):
    """Figure 1: boxes joined by arrows, with labelled brackets under groups of boxes."""

    GAP = 12.0
    PAD = 4.0
    BRACKET = 10.0

    def __init__(self, boxes: list[str], groups: list[tuple[int, int, str]], style):
        super().__init__()
        self.style = style
        self.boxes = [[part.strip() for part in b.split("|")] for b in boxes]
        self.groups = groups
        n = len(self.boxes)
        self.box_w = (MEASURE - self.GAP * (n - 1)) / n
        inner = self.box_w - 2 * self.PAD
        self.paras = []
        for lines in self.boxes:
            p = Paragraph("<br/>".join(html.escape(line) for line in lines), style)
            p.wrap(inner, 1000)
            self.paras.append(p)
        self.box_h = max(p.height for p in self.paras) + 2 * self.PAD + 2
        self.width = MEASURE
        self.height = self.box_h + self.BRACKET + FIGURE_LEADING + 8

    def wrap(self, avail_w, avail_h):
        return self.width, self.height

    def draw(self) -> None:
        c = self.canv
        c.saveState()
        c.setStrokeColor(black)
        c.setFillColor(black)
        c.setLineWidth(RULE_THIN)
        top = self.height
        bottom = top - self.box_h
        for k, para in enumerate(self.paras):
            x = k * (self.box_w + self.GAP)
            c.rect(x, bottom, self.box_w, self.box_h, stroke=1, fill=0)
            para.drawOn(c, x + self.PAD, bottom + (self.box_h - para.height) / 2)
            if k:
                y = bottom + self.box_h / 2
                x0, x1 = x - self.GAP + 2, x - 2
                c.line(x0, y, x1 - 3, y)
                path = c.beginPath()
                path.moveTo(x1, y)
                path.lineTo(x1 - 4.5, y + 2.5)
                path.lineTo(x1 - 4.5, y - 2.5)
                path.close()
                c.drawPath(path, stroke=0, fill=1)
        y = bottom - self.BRACKET + 4
        c.setFont("TNR", FIGURE_TEXT)
        for first, last, label in self.groups:
            x0 = (first - 1) * (self.box_w + self.GAP)
            x1 = (last - 1) * (self.box_w + self.GAP) + self.box_w
            c.line(x0, y, x1, y)
            c.line(x0, y, x0, y + 3)
            c.line(x1, y, x1, y + 3)
            c.drawCentredString((x0 + x1) / 2, y - FIGURE_LEADING + 1, label)
        c.restoreState()


def column_widths(rows: list[list[str]], total: float, pad: float) -> list[float]:
    """Natural widths when they fit; otherwise shrink the wordy columns, never below a word."""
    n = len(rows[0])
    natural, minimum = [0.0] * n, [0.0] * n
    for row in rows:
        for c, cell in enumerate(row):
            bold = cell.startswith("**")
            font = "TNR-Bold" if bold else "TNR"
            text = plain(cell)
            natural[c] = max(natural[c], pdfmetrics.stringWidth(text, font, SMALL) + pad)
            for word in text.split():
                width = pdfmetrics.stringWidth(word, font, SMALL) + pad
                minimum[c] = max(minimum[c], width)
    if sum(minimum) > total:
        raise SystemExit(f"build_pdf: a table's longest words need {sum(minimum):.0f} pt")
    if sum(natural) <= total:
        spare = total - sum(natural)
        return [w + spare * w / sum(natural) for w in natural]
    # Water-filling: a column narrower than its fair share keeps its natural width (no wrap);
    # the wordy columns share what is left in proportion to their natural widths.
    widths: list[float | None] = [None] * n
    budget = total
    while True:
        open_cols = [c for c in range(n) if widths[c] is None]
        fair = budget / len(open_cols)
        fixed = [c for c in open_cols if natural[c] <= fair]
        if not fixed:
            break
        for c in fixed:
            widths[c] = natural[c]
            budget -= natural[c]
    open_cols = [c for c in range(n) if widths[c] is None]
    share = sum(natural[c] for c in open_cols)
    for c in open_cols:
        widths[c] = max(minimum[c], budget * natural[c] / share)
    excess = sum(widths) - total
    if excess > 0:  # a minimum pushed the sum over: take it back from the widest column
        widest = max(open_cols, key=lambda c: widths[c])
        widths[widest] -= excess
    return [float(w) for w in widths]


def table_flowable(block: Block, st: dict[str, ParagraphStyle], pages: dict[str, int]):
    pad = 8.0
    rows = [[substitute(c, pages) for c in row] for row in block.rows]
    widths = column_widths(rows, MEASURE, pad)
    data = []
    for row in rows:
        data.append(
            [
                Paragraph(markup(cell), st["cell_right" if block.align[c] == "right" else "cell"])
                for c, cell in enumerate(row)
            ]
        )
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("LINEABOVE", (0, 0), (-1, 0), RULE_TABLE, black),
                ("LINEBELOW", (0, 0), (-1, 0), RULE_THIN, black),
                ("LINEBELOW", (0, -1), (-1, -1), RULE_TABLE, black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, -1), "TNR"),
                ("LEFTPADDING", (0, 0), (-1, -1), pad / 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), pad / 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ]
        )
    )
    parts = [Paragraph(markup(substitute(block.caption, pages)), st["caption"]), table]
    if block.note:
        parts.append(Paragraph(markup(substitute(block.note, pages)), st["note"]))
    return Group(parts)


def code_flowable(block: Block, st: dict[str, ParagraphStyle]):
    rule_pad = 8.0
    for line in block.items:
        width = pdfmetrics.stringWidth(line, "CourierNew", CODE)
        if width > MEASURE - rule_pad:
            raise SystemExit(f"build_pdf: command line too wide ({width:.0f} pt): {line}")
    pre = XPreformatted(html.escape("\n".join(block.items)), st["code"])
    box = Table([[pre]], colWidths=[MEASURE], hAlign="LEFT")
    box.setStyle(
        TableStyle(
            [
                ("LINEBEFORE", (0, 0), (0, 0), RULE_THIN, black),
                ("FONTNAME", (0, 0), (-1, -1), "TNR"),
                ("LEFTPADDING", (0, 0), (0, 0), rule_pad),
                ("RIGHTPADDING", (0, 0), (0, 0), 0),
                ("TOPPADDING", (0, 0), (0, 0), 1),
                ("BOTTOMPADDING", (0, 0), (0, 0), 1),
            ]
        )
    )
    return [Spacer(1, 2), box, Spacer(1, 6)]


def list_flowables(block: Block, st, pages) -> list:
    out = []
    for k, item in enumerate(block.items, start=1):
        bullet = "•" if block.kind == "bullets" else f"{k}."
        out.append(Paragraph(markup(substitute(item, pages)), st["item"], bulletText=bullet))
    out.append(Spacer(1, 3))
    return out


def summary_flowable(block: Block, st, pages):
    inner = []
    for child in block.children:
        if child.kind == "para":
            inner.append(Paragraph(markup(substitute(child.text, pages)), st["body"]))
        else:
            inner.extend(list_flowables(child, st, pages))
    if isinstance(inner[-1], Spacer):
        inner.pop()
    box = Table([[inner]], colWidths=[MEASURE], hAlign="LEFT")
    box.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), RULE_BOX, black),
                ("FONTNAME", (0, 0), (-1, -1), "TNR"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return [box, Spacer(1, 6)]


def story_for(doc: Document, st, pages: dict[str, int], first: bool) -> list:
    lang = doc.one("lang")
    story: list = []
    if not first:
        story += [NextPageTemplate(lang), PageBreak()]
    anchor = doc.one("anchor")
    story.append(outlined(markup(doc.one("title")), st["title"], anchor, doc.one("outline"), 0))
    story.append(Spacer(1, 4))
    for line in doc.meta.get("meta", []):
        story.append(Paragraph(markup(substitute(line, pages)), st["meta"]))
    target, _, text = doc.one("crosslink").partition("|")
    cross = link(f"#{target.strip()}", markup(substitute(text.strip(), pages)))
    story.append(Paragraph(cross, st["meta"]))
    story.append(Spacer(1, 12))
    heading_no = 0
    for block in doc.blocks:
        if block.kind == "heading":
            heading_no += 1
            text = substitute(block.text, pages)
            key = f"{anchor}-{heading_no}"
            story.append(outlined(markup(text), st["heading"], key, plain(text), 1))
        elif block.kind == "para":
            story.append(body(markup(substitute(block.text, pages)), st["body"]))
        elif block.kind in ("bullets", "numbers"):
            story.extend(list_flowables(block, st, pages))
        elif block.kind == "table":
            story.append(table_flowable(block, st, pages))
        elif block.kind == "code":
            story.extend(code_flowable(block, st))
        elif block.kind == "summary":
            story.extend(summary_flowable(block, st, pages))
        elif block.kind == "figure":
            figure = PipelineFigure(block.items, block.groups, st["figure"])
            caption = Paragraph(markup(block.caption), st["caption"])
            story.append(Group([figure, caption], space_after=8, floating=False))
    while isinstance(story[-1], Spacer):  # a trailing spacer must not open an empty page
        story.pop()
    return story


# ---------------------------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------------------------


class Builder(BaseDocTemplate):
    def __init__(self, path, docs: list[Document], total: int, **kw):
        super().__init__(
            str(path),
            pagesize=A4,
            leftMargin=MARGIN_LR,
            rightMargin=MARGIN_LR,
            topMargin=MARGIN_TB,
            bottomMargin=MARGIN_TB,
            title=PDF_TITLE,
            author=PDF_AUTHOR,
            subject=PDF_SUBJECT,
            creator="docs/technical_report/build_pdf.py",
            producer="docs/technical_report/build_pdf.py",
            keywords=["ball possession", "tracking", "spellF1", "More than a Hack 2027"],
            invariant=1,
            initialFontName="TNR",
            initialFontSize=BODY,
            initialLeading=BODY_LEADING,
            lang="en",
            **kw,
        )
        self.total = total
        self.anchor_pages: dict[str, int] = {}
        frame = Frame(
            MARGIN_LR,
            MARGIN_TB,
            MEASURE,
            PAGE_H - 2 * MARGIN_TB,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id="body",
        )
        self.addPageTemplates(
            [PageTemplate(id=d.one("lang"), frames=[frame], onPage=self.decorator(d)) for d in docs]
        )

    def decorator(self, doc: Document):
        header, footer = doc.one("header"), doc.one("footer")

        def decorate(canvas, _doc):
            canvas.saveState()
            canvas.setFillColor(black)
            canvas.setStrokeColor(black)
            canvas.setFont("TNR", RUNNING)
            canvas.drawString(MARGIN_LR, HEADER_BASELINE, header)
            canvas.setLineWidth(RULE_THIN)
            rule_y = HEADER_BASELINE - 4
            canvas.line(MARGIN_LR, rule_y, PAGE_W - MARGIN_LR, rule_y)
            text = footer.format(page=canvas.getPageNumber(), total=self.total or "?")
            canvas.drawRightString(PAGE_W - MARGIN_LR, FOOTER_BASELINE, text)
            canvas.restoreState()

        return decorate

    def handle_flowable(self, flowables):
        first = flowables[0]
        frame = getattr(self, "frame", None)
        if isinstance(first, Group) and first.floating and frame is not None and not frame._atTop:
            room = frame._y - frame._y1p - first.getSpaceBefore()
            _, height = first.wrap(frame._getAvailableWidth(), room)
            if height > room:
                end = 1
                while end < len(flowables) and getattr(flowables[end], "pullable", False):
                    end += 1
                flowables[0:end] = flowables[1:end] + [first]
        super().handle_flowable(flowables)

    def afterFlowable(self, flowable):
        info = getattr(flowable, "outline", None)
        if info:
            key, title, level = info
            self.anchor_pages[key] = self.page
            self.canv.addOutlineEntry(title, key, level)


def build(docs: list[Document], path: Path) -> tuple[int, dict[str, int]]:
    # Plain Flate streams, not ASCII85: ASCII85 text can spell an e-mail-like string, which the
    # release audit (scripts/release_snapshot.py) rejects. It also keeps the file smaller.
    rl_config.useA85 = 0
    st = styles()
    pages = {"en": 1, "es": 0}
    total = 0
    for _ in range(4):
        story = []
        for k, doc in enumerate(docs):
            story += story_for(doc, st, pages, first=k == 0)
        builder = Builder(path, docs, total)
        builder.build(story)
        new_pages = {d.one("anchor"): builder.anchor_pages[d.one("anchor")] for d in docs}
        new_total = builder.page
        if new_pages == pages and new_total == total:
            return total, pages
        pages, total = new_pages, new_total
    raise SystemExit("build_pdf: page numbers did not settle")


# ---------------------------------------------------------------------------------------------
# Style-guide checks (--check)
# ---------------------------------------------------------------------------------------------

WORD = re.compile(r"[0-9A-Za-zÀ-ÿ]")
ABBREVIATIONS = ("et al.", "e.g.", "i.e.", "p. ej.")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡0-9(+−±])")
NUMBER = re.compile(r"[+−±-]?\d+(?:[.,]\d+)?")


def count_words(text: str) -> int:
    return sum(1 for token in text.split() if WORD.search(token))


def sentences(text: str) -> list[str]:
    protected = text
    for abbr in ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", "\u0000"))
    return [s.replace("\u0000", ".") for s in SENTENCE_END.split(protected) if s.strip()]


def lead_in(text: str) -> tuple[str, str]:
    """The paragraph's bold run-in lead-in and the text after it ("" when there is none)."""
    m = re.match(r"\*\*(.+?)\*\*\s*", text)
    if m and m.group(1).rstrip().endswith((".", ":")):
        return m.group(1), text[m.end() :]
    return "", text


def all_text(doc: Document) -> list[str]:
    out = [doc.one("title"), *doc.meta.get("meta", []), doc.one("crosslink").partition("|")[2]]

    def walk(blocks):
        for b in blocks:
            if b.kind in ("heading", "para"):
                out.append(b.text)
            elif b.kind in ("bullets", "numbers", "code"):
                out.extend(b.items)
            elif b.kind == "table":
                out.append(b.caption)
                out.extend(" ".join(row) for row in b.rows)
                out.append(b.note)
            elif b.kind == "figure":
                out.extend(i.replace("|", " ") for i in b.items)
                out.extend(g[2] for g in b.groups)
                out.append(b.caption)
            elif b.kind == "summary":
                walk(b.children)

    walk(doc.blocks)
    return [plain(t) for t in out if t]


def prose_units(doc: Document) -> list[str]:
    """Paragraphs and list items, summary included: the units the sentence rules apply to."""
    units: list[str] = []

    def walk(blocks):
        for b in blocks:
            if b.kind == "para":
                units.append(b.text)
            elif b.kind in ("bullets", "numbers"):
                units.extend(b.items)
            elif b.kind == "summary":
                walk(b.children)

    walk(doc.blocks)
    return units


def skim_layer(doc: Document, with_summary: bool = True) -> str:
    """Headings, every bold span, and each section's first sentence (after its lead-in)."""
    parts: list[str] = []
    first_after_heading = False

    def bold_spans(text: str) -> list[str]:
        return [plain(m) for m in BOLD.findall(text)]

    def walk(blocks):
        nonlocal first_after_heading
        for b in blocks:
            if b.kind == "heading":
                parts.append(plain(b.text))
                first_after_heading = True
            elif b.kind == "para":
                lead, rest = lead_in(b.text)
                if first_after_heading:
                    first = sentences(rest)[0] if rest.strip() else ""
                    parts.append(plain(lead))
                    parts.append(plain(first))
                    tail = rest[len(first) :]
                    parts.extend(bold_spans(tail))
                    first_after_heading = False
                else:
                    parts.extend(bold_spans(b.text))
            elif b.kind in ("bullets", "numbers"):
                for item in b.items:
                    parts.extend(bold_spans(item))
            elif b.kind == "table":
                parts.extend(bold_spans(b.caption))
                for row in b.rows:
                    for cell in row:
                        parts.extend(bold_spans(cell))
            elif b.kind == "figure":
                parts.extend(bold_spans(b.caption))
            elif b.kind == "summary" and with_summary:
                for child in b.children:
                    for unit in [child.text, *child.items]:
                        parts.extend(bold_spans(unit))

    walk(doc.blocks)
    return " ".join(p for p in parts if p)


def number_in(number: str, text: str) -> bool:
    bare = number.lstrip("+−±-")
    pattern = r"(?<![\w.,])[+−±-]?" + re.escape(bare) + r"(?![\w]|[.,]\d)"
    return re.search(pattern, text) is not None


def check(doc: Document) -> list[str]:
    lang = doc.one("lang")
    texts = all_text(doc)
    total = sum(count_words(t) for t in texts)
    units = prose_units(doc)
    bold_words = 0

    def walk_bold(blocks):
        nonlocal bold_words
        for b in blocks:
            for t in [b.text, b.caption, *b.items, *(" ".join(r) for r in b.rows)]:
                bold_words += sum(count_words(plain(m)) for m in BOLD.findall(t))
            walk_bold(b.children)

    walk_bold(doc.blocks)
    lengths: list[int] = []
    long_paras = no_lead = long_leads = heavy = 0
    for unit in units:
        lead, rest = lead_in(unit)
        sents = sentences(rest) if rest.strip() else []
        lengths += [count_words(plain(s)) for s in sents]
        if count_words(plain(unit)) > 80 or len(sents) > 4:
            long_paras += 1
        if not lead:
            no_lead += 1
        elif count_words(lead) > 6:
            long_leads += 1
        if sum(1 for m in BOLD.findall(rest) if NUMBER.search(m)) > 2:
            heavy += 1
    skim = skim_layer(doc)
    body_skim = skim_layer(doc, with_summary=False)
    lines = [
        f"[{lang}] words: {total} in all (tables, commands, captions, title block included)",
        f"[{lang}] bold: {bold_words} words, {100 * bold_words / total:.1f}% of all words",
        f"[{lang}] sentences: {len(lengths)}, mean {sum(lengths) / len(lengths):.1f} words, "
        f"max {max(lengths)}, over 25 words: {sum(n > 25 for n in lengths)}",
        f"[{lang}] paragraphs and items: {len(units)}; over 80 words or 4 sentences: {long_paras}; "
        f"without a bold lead-in: {no_lead}; lead-ins over 6 words: {long_leads}; "
        f"over 2 bold numbers: {heavy}",
        f"[{lang}] skim layer (headings + bold + each section's first sentence): "
        f"{count_words(skim)} words",
    ]
    summary = next(b for b in doc.blocks if b.kind == "summary")
    claims = [item for child in summary.children for item in child.items]
    for k, claim in enumerate(claims, start=1):
        _, rest = lead_in(claim)
        numbers = NUMBER.findall(plain(rest))
        missing = [n for n in numbers if not number_in(n, body_skim)]
        terms = re.findall(r"[A-Za-zÀ-ÿ]{5,}", plain(rest))
        found = [t for t in terms if t.lower() in body_skim.lower()]
        lines.append(
            f"[{lang}] summary claim {k} against the skim layer outside the Summary: numbers "
            f"{len(numbers) - len(missing)}/{len(numbers)}"
            + (f" (missing {missing})" if missing else "")
            + f", words of 5+ letters {len(found)}/{len(terms)}"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="print the style-guide checks")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    register_fonts()
    docs = [parse(path.read_text(encoding="utf-8")) for path in SOURCES]
    total, pages = build(docs, args.out)
    print(f"wrote {args.out.name}: {total} pages; Spanish starts on page {pages['es']}")
    if args.check:
        for doc in docs:
            print("\n".join(check(doc)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
