"""Build every lesson (slides, notebook, study sheet) and the course index.
Usage: python build/build_all.py [w1 w2 ...]"""
import importlib
import sys
import time

from framework import ROOT, write_lesson

weeks = sys.argv[1:] or ["w1", "w2", "w3", "w4"]
for w in weeks:
    try:
        mod = importlib.import_module(w)
    except ModuleNotFoundError:
        print("skip", w)
        continue
    for lesson in mod.LESSONS:
        t0 = time.time()
        d = write_lesson(lesson)
        print(f"{d.relative_to(ROOT)}  ({time.time() - t0:.1f}s, {len(lesson.slides) + 1} slides, {len(lesson.cells)} cells)")

# ---------------------------------------------------------------- course pages
from framework import render_doc_page, render_index

all_lessons = []
for w in ["w1", "w2", "w3", "w4"]:
    try:
        all_lessons += importlib.import_module(w).LESSONS
    except ModuleNotFoundError:
        pass
(ROOT / "index.html").write_text(render_index(all_lessons), encoding="utf-8")
(ROOT / "coursework.html").write_text(
    render_doc_page((ROOT / "COURSEWORK.md").read_text(encoding="utf-8"), "期末作业"), encoding="utf-8")
print("index.html and coursework.html written for", len(all_lessons), "lessons")
