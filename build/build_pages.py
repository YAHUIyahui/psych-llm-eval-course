"""Rebuild slides, study sheets and course pages without touching notebooks."""
import importlib
from framework import ROOT, cjk_punct, render_doc_page, render_index, render_sheet, render_slides

lessons = []
for w in ["w1", "w2", "w3", "w4"]:
    lessons += importlib.import_module(w).LESSONS
for l in lessons:
    d = ROOT / "lessons" / l.folder
    (d / "slides.html").write_text(render_slides(l), encoding="utf-8")
    (d / "README.md").write_text(cjk_punct(render_sheet(l)), encoding="utf-8")
(ROOT / "index.html").write_text(render_index(lessons), encoding="utf-8")
(ROOT / "coursework.html").write_text(render_doc_page((ROOT / "COURSEWORK.md").read_text(encoding="utf-8"), "期末作业"), encoding="utf-8")
print("pages rebuilt for", len(lessons), "lessons")
