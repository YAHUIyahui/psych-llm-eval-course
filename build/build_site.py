"""Build a publishable copy of the course pages (index + slides + coursework)."""
import importlib
import sys
from pathlib import Path

from framework import ROOT, render_doc_page, render_index, render_slides

out = Path(sys.argv[1])
lessons = []
for w in ["w1", "w2", "w3", "w4"]:
    lessons += importlib.import_module(w).LESSONS
for l in lessons:
    d = out / "lessons" / l.folder
    d.mkdir(parents=True, exist_ok=True)
    (d / "slides.html").write_text(render_slides(l, home_href="../../"), encoding="utf-8")
(out / "index.html").write_text(render_index(lessons, site=True), encoding="utf-8")
(out / "coursework.html").write_text(render_doc_page((ROOT / "COURSEWORK.md").read_text(encoding="utf-8"), "期末作业", home="./"), encoding="utf-8")
print("site written to", out)
