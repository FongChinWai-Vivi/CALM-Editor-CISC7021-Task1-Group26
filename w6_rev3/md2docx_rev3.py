# -*- coding: utf-8 -*-
"""Render W6_rev3_measured_e2e.md to .docx (tables preserved as real tables)."""
import os
import re

from docx import Document
from docx.shared import Pt

REPO = r"C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26"
MD = os.path.join(REPO, "FONG CHIN WAI", "w6_deferred", "W6_rev3_measured_e2e.md")
OUT = os.path.join(REPO, "FONG CHIN WAI", "w6_deferred", "W6_rev3_measured_e2e.docx")
BOLD = re.compile(r"\*\*(.+?)\*\*")
INLINE = re.compile(r"`([^`]+)`")


def plain(s):
    return INLINE.sub(r"\1", BOLD.sub(r"\1", s))


doc = Document()
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(10.5)
lines = open(MD, encoding="utf-8").read().split("\n")
i = 0
while i < len(lines):
    ln = lines[i]
    if ln.startswith("```"):
        j = i + 1
        buf = []
        while j < len(lines) and not lines[j].startswith("```"):
            buf.append(lines[j])
            j += 1
        p = doc.add_paragraph("\n".join(buf))
        for r in p.runs:
            r.font.name = "Consolas"
            r.font.size = Pt(8.5)
        i = j + 1
        continue
    if ln.startswith("|") and i + 1 < len(lines) and \
            set(lines[i + 1].replace("|", "").replace(" ", "")) <= set("-:"):
        hdr = [plain(c.strip()) for c in ln.strip().strip("|").split("|")]
        rows = []
        j = i + 2
        while j < len(lines) and lines[j].startswith("|"):
            rows.append([plain(c.strip()) for c in lines[j].strip().strip("|").split("|")])
            j += 1
        t = doc.add_table(rows=1, cols=len(hdr))
        t.style = "Table Grid"
        for k, h in enumerate(hdr):
            t.rows[0].cells[k].text = h
        for row in rows:
            cells = t.add_row().cells
            for k, c in enumerate(row[:len(hdr)]):
                cells[k].text = c
        for r in t.rows:
            for c in r.cells:
                for p in c.paragraphs:
                    for run in p.runs:
                        run.font.size = Pt(8.5)
        doc.add_paragraph("")
        i = j
        continue
    if ln.startswith("# "):
        doc.add_heading(plain(ln[2:]), 0)
    elif ln.startswith("## "):
        doc.add_heading(plain(ln[3:]), 1)
    elif ln.startswith("### "):
        doc.add_heading(plain(ln[4:]), 2)
    elif ln.startswith("> "):
        p = doc.add_paragraph(plain(ln[2:]))
        p.paragraph_format.left_indent = Pt(18)
        for r in p.runs:
            r.italic = True
    elif ln.strip().startswith(("- ", "* ")):
        doc.add_paragraph(plain(ln.strip()[2:]), style="List Bullet")
    elif re.match(r"^\d+\. ", ln.strip()):
        doc.add_paragraph(plain(re.sub(r"^\d+\. ", "", ln.strip())), style="List Number")
    elif ln.strip() == "---":
        doc.add_paragraph("_" * 62)
    elif ln.strip():
        doc.add_paragraph(plain(ln.strip()))
    i += 1

doc.save(OUT)
print("saved", OUT, os.path.getsize(OUT))
