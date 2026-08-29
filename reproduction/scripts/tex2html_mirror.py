#!/usr/bin/env python
"""Render the paper as a single self-contained HTML page for review.

    python scripts/tex2html_mirror.py [out.html]

Follows paper/main.tex through its \\input sections, resolves the
result macros, embeds every referenced figure as a PNG data URI, and
renders tables with the paper's emphasis conventions (bold = best,
gray = within one sd, underline = tradeoff column styling is kept as
produced by the generators). A review mirror, deliberately simpler
than LaTeX: unknown constructs degrade to readable text.
"""

from __future__ import annotations

import base64
import html
import io
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"

# ---------------------------------------------------------------- inputs
macros: dict[str, str] = {}
for m in re.finditer(r"\\newcommand\{\\(Res\w+)\}\{([^{}]*)\}",
                     (PAPER / "results_macros.tex").read_text()):
    macros[m.group(1)] = m.group(2)

cites: dict[str, str] = {}
_bib = (PAPER / "references.bib").read_text()
for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", _bib, re.S):
    key, body = m.group(1).strip(), m.group(2)
    am = re.search(r"author\s*=\s*[{\"](.*?)[}\"],?\n", body, re.S)
    ym = re.search(r"year\s*=\s*[{\"]?(\d{4})", body)
    if not (am and ym):
        cites[key] = key
        continue
    first = am.group(1).split(" and ")[0]
    surname = first.split(",")[0].strip() if "," in first \
        else first.split()[-1]
    n = len(am.group(1).split(" and "))
    cites[key] = surname + (" et al." if n > 2 else "") + ", " + ym.group(1)


def balanced(text: str, command: str) -> list[tuple[int, str, int]]:
    """(start, body, end) for every \\command{...} with nested braces."""
    out = []
    for m in re.finditer(r"\\" + command + r"\{", text):
        depth, i = 1, m.end()
        while depth and i < len(text):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        out.append((m.start(), text[m.end():i - 1], i))
    return out


# ------------------------------------------------------- label numbering
labels: dict[str, str] = {}


def collect_labels(chunks: list[tuple[str, str]]) -> None:
    """Assign display numbers to every \\label in document order."""
    sec = 0
    sub = 0
    fig = 0
    tab = 0
    app = 0
    in_appendix = False
    for kind, text in chunks:
        if kind == "appendix-marker":
            in_appendix = True
            continue
        pos_events = []
        for m in re.finditer(r"\\(section|subsection)\{", text):
            pos_events.append((m.start(), m.group(1)))
        for env in ("figure", "table"):
            for m in re.finditer(r"\\begin\{" + env + r"\}", text):
                pos_events.append((m.start(), env))
        pos_events.sort()
        # walk labels and attribute to the nearest preceding event
        for lm in re.finditer(r"\\label\{([^}]*)\}", text):
            ev = [e for e in pos_events if e[0] < lm.start()]
            kind2 = ev[-1][1] if ev else "section"
            key = lm.group(1)
            if key in labels:
                continue
            # count events up to here to number deterministically
            n_before = sum(1 for e in ev if e[1] == kind2)
            if kind2 == "section":
                if in_appendix:
                    labels[key] = f"Appendix {chr(64 + app + n_before)}"
                else:
                    labels[key] = f"Section {sec + n_before}"
            elif kind2 == "subsection":
                labels[key] = "Section"  # refined below if needed
                labels[key] = f"Section {sec + max(1, 0)}.{n_before}" \
                    if not in_appendix else \
                    f"Appendix {chr(64 + max(app, 1))}.{n_before}"
            elif kind2 == "figure":
                labels[key] = f"Figure {fig + n_before}"
            else:
                labels[key] = f"Table {tab + n_before}"
        sec += sum(1 for _, k in pos_events
                   if k == "section" and not in_appendix)
        app += sum(1 for _, k in pos_events
                   if k == "section" and in_appendix)
        fig += sum(1 for _, k in pos_events if k == "figure")
        tab += sum(1 for _, k in pos_events if k == "table")


def cref(key: str) -> str:
    name = labels.get(key.strip(), key.strip())
    return f'<a href="#{html.escape(key.strip())}">{name}</a>'


# ---------------------------------------------------------------- prose
def mathspan(s: str) -> str:
    s = s.replace("\\pm", "±").replace("\\times", "×")
    s = s.replace("\\sim", "~").replace("\\to", "→")
    s = s.replace("\\le", "≤").replace("\\ge", "≥")
    s = s.replace("\\sqrt", "√").replace("\\rho", "ρ")
    s = s.replace("\\Delta", "Δ").replace("\\phi", "φ")
    s = s.replace("\\emptyset", "∅").replace("\\in", "∈")
    s = s.replace("\\mathcal{C}", "𝒞").replace("\\mathcal{X}", "𝒳")
    s = s.replace("\\mathrm", "").replace("\\text", "")
    s = re.sub(r"\^\{([^{}]*)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"\^(\w|\*)", r"<sup>\1</sup>", s)
    s = re.sub(r"_(\w|\*)", r"<sub>\1</sub>", s)
    s = s.replace("{", "").replace("}", "")
    return f'<span class="math">{s}</span>'


def prose(s: str) -> str:
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = s.replace("\\method{}", "TopoSHAP").replace("\\method", "TopoSHAP")
    s = re.sub(r"\\unfrozen\{\\(Res\w+)\}",
               lambda m: f'<span class="uf">'
                         f'{macros.get(m.group(1), m.group(1))}</span>', s)
    s = re.sub(r"\\unfrozen\{([^{}]*)\}", r'<span class="uf">\1</span>', s)
    s = re.sub(r"\\(Res\w+)", lambda m: macros.get(m.group(1),
                                                   m.group(0)), s)
    s = re.sub(r"\{\\scriptsize\s*([^{}]*)\}", r"<small>\1</small>", s)
    s = re.sub(r"\{\\small\s*([^{}]*)\}", r"<small>\1</small>", s)
    s = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[Cc]ref\{([^}]*)\}",
               lambda m: ", ".join(cref(k) for k in m.group(1).split(",")),
               s)
    s = re.sub(r"\\citep\{([^}]*)\}",
               lambda m: "(" + "; ".join(cites.get(k.strip(), k)
                                         for k in m.group(1).split(","))
               + ")", s)
    s = re.sub(r"\\citet\{([^}]*)\}",
               lambda m: "; ".join(cites.get(k.strip(), k)
                                   for k in m.group(1).split(",")), s)
    s = re.sub(r"\\textbf\{([^{}]*)\}", r"<b>\1</b>", s)
    s = re.sub(r"\\emph\{([^{}]*)\}", r"<em>\1</em>", s)
    s = re.sub(r"\\texttt\{([^{}]*)\}", r"<code>\1</code>", s)
    s = re.sub(r"\\underline\{([^{}]*)\}", r"<u>\1</u>", s)
    s = re.sub(r"\\nbhd\{([^{}]*)\}", r"<code>\1</code>", s)
    s = re.sub(r"\\url\{([^{}]*)\}", r"<code>\1</code>", s)
    s = re.sub(r"\\footnote\{([^{}]*)\}", r" <small>[\1]</small>", s)
    s = re.sub(r"\$([^$]+)\$", lambda m: mathspan(m.group(1)), s)
    s = s.replace("``", "\u201c").replace("''", "\u201d")
    s = s.replace("\\%", "%").replace("\\&", "&amp;").replace("\\_", "_")
    s = s.replace("~", " ").replace("\\,", " ").replace("\\;", " ")
    s = s.replace("\\eg", "e.g.").replace("\\ie", "i.e.")
    s = s.replace("---", " — ").replace("--", "–")
    s = s.replace("\\{", "{").replace("\\}", "}")
    s = re.sub(r"\\(todoresearcher(?:inline)?)\{([^{}]*)\}",
               r'<span class="todo">TODO(researcher): \2</span>', s)
    return s


# ---------------------------------------------------------------- tables
def render_cell(c: str, tag: str) -> str:
    span = ""
    group = False
    mc = re.match(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}$", c)
    if mc:
        span = f' colspan="{mc.group(1)}"'
        c = mc.group(2)
        group = tag == "td"
    near = "\\cellcolor" in c
    c = re.sub(r"\\cellcolor\{[^}]*\}", "", c).strip()
    c = re.sub(r"\\unfrozen\{([^{}]*)\}", r'<span class="uf">\1</span>', c)
    deal = "\\underline{" in c
    bold = "\\textbf{" in c
    c = re.sub(r"\\textbf\{(.*?)\}", r"\1", c)
    c = re.sub(r"\\underline\{(.*?)\}", r"\1", c)
    c = prose(c)
    classes = [x for x, ok in (("best", bold), ("near", near),
                               ("deal", deal), ("group", group)) if ok]
    cls = f' class="{" ".join(classes)}"' if classes else ""
    return f"<{tag}{cls}{span}>{c}</{tag}>"


def tex_table(block: str) -> str:
    caps = balanced(block, "caption")
    cap = caps[0][1] if caps else ""
    lm = re.search(r"\\label\{([^}]*)\}", block)
    lab = lm.group(1) if lm else ""
    try:
        body = block[block.index("\\midrule"):block.index("\\bottomrule")]
        head = block[block.index("\\toprule"):block.index("\\midrule")]
    except ValueError:
        return f'<figure class="tbl" id="{lab}">' \
               f"<figcaption>{prose(cap)}</figcaption></figure>"

    def rows_of(chunk: str, tag: str) -> list[str]:
        out = []
        for line in chunk.splitlines():
            line = line.strip()
            if not line.endswith("\\\\") or "\\cmidrule" in line:
                continue
            cells = [c.strip() for c in line[:-2].split("&")]
            tr = ["<tr>"]
            for j, c in enumerate(cells):
                t = tag if tag == "th" else (
                    "th" if j == 0 and len(cells) > 1 else "td")
                tr.append(render_cell(c, t))
            tr.append("</tr>")
            out.append("".join(tr))
        return out

    parts = [f'<figure class="tbl" id="{lab}">',
             f"<figcaption>{prose(cap)}</figcaption>",
             "<div class='tablewrap'><table><thead>",
             *rows_of(head, "th"),
             "</thead><tbody>",
             *rows_of(body, "td"),
             "</tbody></table></div></figure>"]
    return "\n".join(parts)


# --------------------------------------------------------------- figures
def png_uri(pdf: Path, dpi: int = 95) -> str | None:
    try:
        import pymupdf
        pix = pymupdf.open(pdf)[0].get_pixmap(dpi=dpi)
        return "data:image/png;base64," + base64.b64encode(
            pix.tobytes("png")).decode()
    except Exception:
        return None


def tex_figure(block: str) -> str:
    caps = balanced(block, "caption")
    cap = caps[0][1] if caps else ""
    lm = re.search(r"\\label\{([^}]*)\}", block)
    lab = lm.group(1) if lm else ""
    im = re.search(r"\\includegraphics\[[^]]*\]\{([^}]*)\}", block)
    img = ""
    if im:
        p = (PAPER / im.group(1)).resolve()
        uri = png_uri(p) if p.exists() else None
        img = (f'<img src="{uri}" alt="">' if uri else
               f'<div class="missing">figure pending: {p.name}</div>')
    return (f'<figure id="{lab}">{img}'
            f"<figcaption>{prose(cap)}</figcaption></figure>")


# ------------------------------------------------------------ conversion
ENV_BOXES = {"recipebox": "Practitioner's recipe.",
             "theorybox": ""}


def convert(text: str) -> str:
    out = []
    # cut out figure/table/equation environments first, replace by tokens
    tokens: list[str] = []

    def stash(html_: str) -> str:
        tokens.append(html_)
        return f"\n@@TOK{len(tokens) - 1}@@\n"

    def items(s: str) -> str:
        if "\\item" not in s:
            return s
        parts = re.split(r"\\item\s*", s)
        lead = parts[0]
        lis = "".join(f"<li>{p.strip()}</li>" for p in parts[1:] if p.strip())
        return f"{lead}<ol>{lis}</ol>"

    for env, fn in (("figure", tex_figure), ("table", tex_table)):
        pat = re.compile(r"\\begin\{" + env + r"\}(?:\[[^]]*\])?(.*?)"
                         r"\\end\{" + env + r"\}", re.S)
        text = pat.sub(lambda m: stash(fn(m.group(1))), text)
    text = re.sub(
        r"\\begin\{equation\}(.*?)\\end\{equation\}", lambda m: stash(
            f'<div class="eq">{mathspan(m.group(1).strip())}</div>'),
        text, flags=re.S)
    for env, title in ENV_BOXES.items():
        text = re.sub(
            r"\\begin\{" + env + r"\}(.*?)\\end\{" + env + r"\}",
            lambda m, t=title: stash(
                f'<div class="box">{"<b>" + t + "</b> " if t else ""}'
                f"{prose(items(m.group(1)))}</div>"), text, flags=re.S)
    for env in ("definition", "proposition", "propositionT",
                "corollaryT", "propositionTrank", "remark"):
        text = re.sub(
            r"\\begin\{" + env + r"\}(?:\[([^]]*)\])?(.*?)\\end\{"
            + env + r"\}",
            lambda m, e=env: stash(
                f'<div class="thm"><b>{e.capitalize()}'
                f"{' (' + m.group(1) + ')' if m.group(1) else ''}.</b> "
                f"{prose(m.group(2))}</div>"), text, flags=re.S)

    text = re.sub(r"\\begin\{(itemize|enumerate)\}(.*?)\\end\{\1\}",
                  lambda m: stash(
                      ("<ul>" if m.group(1) == "itemize" else "<ol>")
                      + "".join(f"<li>{prose(p.strip())}</li>"
                                for p in re.split(r"\\item\s*", m.group(2))
                                if p.strip())
                      + ("</ul>" if m.group(1) == "itemize" else "</ol>")),
                  text, flags=re.S)

    # expclaims (nested braces)
    text = text.replace("\\expclaim", "\\expclaimX")
    while True:
        hits = balanced(text, "expclaimX")
        if not hits:
            break
        s0, body, e0 = hits[0]
        text = text[:s0] + stash(
            f'<p class="claim"><b>{prose(body)}</b></p>') + text[e0:]

    # headings + labels
    text = re.sub(r"\\section\*?\{([^{}]*)\}\s*(?:\\label\{([^}]*)\})?",
                  lambda m: stash(
                      f'<h2 id="{m.group(2) or ""}">{prose(m.group(1))}'
                      "</h2>"), text)
    text = re.sub(r"\\subsection\*?\{([^{}]*)\}\s*(?:\\label\{([^}]*)\})?",
                  lambda m: stash(
                      f'<h3 id="{m.group(2) or ""}">{prose(m.group(1))}'
                      "</h3>"), text)
    text = re.sub(r"\\paragraph\{([^{}]*)\}",
                  lambda m: stash(f"<p><b>{prose(m.group(1))}</b></p>"),
                  text)
    text = re.sub(r"\\label\{([^}]*)\}",
                  lambda m: stash(f'<span id="{m.group(1)}"></span>'), text)

    # paragraphs
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if re.fullmatch(r"(@@TOK\d+@@\s*)+", para):
            out.append(para)
            continue
        out.append(f"<p>{prose(para)}</p>")
    page = "\n".join(out)
    page = re.sub(r"@@TOK(\d+)@@", lambda m: tokens[int(m.group(1))], page)
    return page


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        PAPER / "build" / "mirror.html")
    main_tex = (PAPER / "main.tex").read_text()
    title_m = re.search(r"\\newcommand\{\\papertitle\}\{(.*?)\}\n", main_tex,
                        re.S)
    title = re.sub(r"\s+", " ", title_m.group(1)) if title_m else "TopoSHAP"
    abs_m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_tex,
                      re.S)

    chunks: list[tuple[str, str]] = []
    for line in main_tex.splitlines():
        line = line.strip()
        if line.startswith("\\appendix"):
            chunks.append(("appendix-marker", ""))
        m = re.match(r"\\input\{(sections/[^}]+)\}", line)
        if m:
            chunks.append((m.group(1),
                           (PAPER / f"{m.group(1)}.tex").read_text()))
    # expand table \input inside sections for label collection + render
    expanded = []
    for name, text in chunks:
        text = re.sub(
            r"\\input\{(tables/[^}]+)\}",
            lambda m: (PAPER / f"{m.group(1)}.tex").read_text()
            if (PAPER / f"{m.group(1)}.tex").exists() else "", text)
        expanded.append((name, text))
    collect_labels(expanded)

    body = [f"<h1>{prose(title)}</h1>",
            '<p class="authors">Anonymous authors — draft mirror, '
            "regenerated from paper/ sources</p>"]
    if abs_m:
        body.append(f'<div class="abstract"><b>Abstract.</b> '
                    f"{prose(abs_m.group(1).strip())}</div>")
    for name, text in expanded:
        if name == "appendix-marker":
            body.append("<hr><h1>Appendix</h1>")
            continue
        body.append(convert(text))

    css = """
:root { --bg:#ffffff; --ink:#1a1a1a; --muted:#6a6a6a; --acc:#1f3d99;
  --box:#f4f4f2; --line:#d8d8d4; }
:root:not([data-theme="light"]) { }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#191a1c; --ink:#e8e6e1; --muted:#9a9a94; --acc:#8fa7e8;
  --box:#232427; --line:#3a3b3e; } }
:root[data-theme="dark"] { --bg:#191a1c; --ink:#e8e6e1; --muted:#9a9a94;
  --acc:#8fa7e8; --box:#232427; --line:#3a3b3e; }
body { background:var(--bg); color:var(--ink);
  font:16px/1.55 Georgia, 'Times New Roman', serif;
  max-width:46rem; margin:2rem auto; padding:0 1.2rem; }
h1,h2,h3 { font-family:Helvetica, Arial, sans-serif; line-height:1.25; }
h1 { font-size:1.5rem; } h2 { font-size:1.2rem; margin-top:2.2rem; }
h3 { font-size:1.02rem; margin-top:1.6rem; }
a { color:var(--acc); text-decoration:none; }
.authors { color:var(--muted); font-style:italic; }
.abstract { background:var(--box); border:1px solid var(--line);
  border-radius:8px; padding:1rem 1.2rem; margin:1.4rem 0; }
.box,.thm { background:var(--box); border:1px solid var(--line);
  border-radius:8px; padding:.8rem 1rem; margin:1rem 0; }
.eq { text-align:center; margin:1rem 0; font-style:italic; }
.claim { margin-top:1.3rem; }
.uf { border-bottom:1px dotted var(--muted); }
.todo { color:#b04a30; font-size:.85em; font-weight:600; }
figure { margin:1.6rem 0; }
figure img { max-width:100%; border:1px solid var(--line);
  border-radius:6px; background:#fff; }
figcaption { font-size:.86rem; color:var(--muted); margin-top:.5rem; }
.missing { border:1px dashed var(--line); color:var(--muted);
  padding:1.4rem; text-align:center; border-radius:6px; }
.tablewrap { overflow-x:auto; }
table { border-collapse:collapse; font:12.5px/1.4 Helvetica, Arial,
  sans-serif; margin:.4rem 0; }
th,td { padding:3px 8px; text-align:left; white-space:nowrap; }
thead tr:last-child th { border-bottom:1px solid var(--ink); }
tbody tr:last-child td, tbody tr:last-child th {
  border-bottom:1px solid var(--ink); }
thead tr:first-child th { border-top:1px solid var(--ink); }
td.best, th.best { font-weight:700; }
td.near, th.near { background:color-mix(in srgb, var(--ink) 9%,
  transparent); }
td.deal, th.deal { text-decoration:underline;
  text-underline-offset:3px; }
td.group { font-style:italic; font-weight:600; padding-top:.55em; }
.math { font-style:italic; }
"""
    page = (f"<title>TopoSHAP Draft</title>\n<style>{css}</style>\n"
            + "\n".join(body))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page)
    print(f"wrote {out_path} {len(page) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
