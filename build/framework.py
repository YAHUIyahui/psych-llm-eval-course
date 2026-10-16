"""Course builder: turns lesson definitions into slides.html, notebook.ipynb
and a study sheet per lesson, plus the course index."""
from __future__ import annotations

import base64
import html
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
COURSE_NAME = "心理大模型评测与 Agent"
TRACK_LABEL = {"eval": "评测方法学", "agent": "Agent 与模型", "project": "项目实战", "review": "复习与习题", "career": "求职准备"}


# ------------------------------------------------------------------ helpers for slide bodies
def esc(s):
    return html.escape(s, quote=False)


_CJK = r"\u3000-\u303f\u4e00-\u9fff\uff00-\uffef"
_FULL = {",": "，", ":": "：", ";": "；", "?": "？", "(": "（", ")": "）"}


def cjk_punct(text: str) -> str:
    """Use full-width , : ; ? ( ) next to Chinese characters. Code spans,
    $math$, URLs and markdown link targets are left untouched."""
    parts = re.split(r"(`[^`]*`|\$\$.*?\$\$|\$[^$\n]*\$|https?://\S+|\]\([^)]*\))", text, flags=re.S)
    out = []
    for k, seg in enumerate(parts):
        if k % 2 == 1:
            out.append(seg)
            continue
        chars = list(seg)
        for i, ch in enumerate(chars):
            if ch in _FULL:
                j = i - 1
                while j >= 0 and chars[j] == "*":
                    j -= 1
                k2 = i + 1
                while k2 < len(chars) and chars[k2] == "*":
                    k2 += 1
                prev = chars[j] if j >= 0 else ""
                nxt = chars[k2] if k2 < len(chars) else ""
                if re.match(f"[{_CJK}]", prev) or re.match(f"[{_CJK}]", nxt):
                    chars[i] = _FULL[ch]
        out.append("".join(chars))
    return "".join(out)


def md_inline(s: str) -> str:
    """Tiny inline markup: **bold**, `code`, keep everything else escaped."""
    s = esc(cjk_punct(s))
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


def P(text):
    return f"<p>{md_inline(text)}</p>"


def LEAD(text):
    return f'<p class="lead">{md_inline(text)}</p>'


def UL(items):
    return "<ul class=\"points\">" + "".join(f"<li>{md_inline(i)}</li>" for i in items) + "</ul>"


def OL(items):
    return "<ol class=\"steps\">" + "".join(f"<li>{md_inline(i)}</li>" for i in items) + "</ol>"


def FORMULA(expr, note=""):
    n = f'<span class="fnote">{md_inline(note)}</span>' if note else ""
    return f'<div class="formula"><span class="fexpr">{esc(expr)}</span>{n}</div>'


def NOTE(text, label="要点"):
    return f'<div class="note"><span class="tag">{esc(label)}</span><p>{md_inline(text)}</p></div>'


def WARN(text, label="常见错误"):
    return f'<div class="note warn"><span class="tag">{esc(label)}</span><p>{md_inline(text)}</p></div>'


def CODE(src):
    return f'<pre class="code"><code>{esc(src.strip())}</code></pre>'


def TABLE(header, rows):
    h = "".join(f"<th>{md_inline(c)}</th>" for c in header)
    b = "".join("<tr>" + "".join(f"<td>{md_inline(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table class="t"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def COLS(left, right):
    return f'<div class="cols"><div>{left}</div><div>{right}</div></div>'


def QA(q, a):
    return f'<div class="qa"><p class="q">{md_inline(q)}</p><p class="a">{md_inline(a)}</p></div>'


def FIG(fig_fn, caption=""):
    """Render a matplotlib figure function to an inline PNG plate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from course.plotting import use_style
    use_style()
    fig = fig_fn()
    for ax in fig.axes:
        ax.set_facecolor("#fdfaf5")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#fdfaf5")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    cap = f"<figcaption>{md_inline(caption)}</figcaption>" if caption else ""
    return f'<figure class="plate"><img alt="{esc(caption or "figure")}" src="data:image/png;base64,{b64}">{cap}</figure>'


# ------------------------------------------------------------------ lesson model
@dataclass
class Paper:
    cite: str
    why: str
    questions: list
    link: str = ""


@dataclass
class Lesson:
    day: int
    slug: str
    title: str
    track: str
    goals: list
    slides: list                      # list of (title, html)
    cells: list                       # list of ("md"|"code", text)
    paper: Paper | None = None
    research: str = ""
    interview: list = field(default_factory=list)   # list of (question, key points)
    homework: list = field(default_factory=list)
    hours: str = "3 到 4 小时"
    solutions: dict = field(default_factory=dict)       # answer variable -> python expression
    replacements: list = field(default_factory=list)    # (old code, new code) for function exercises

    @property
    def folder(self):
        return f"day{self.day:02d}_{self.slug}"


# ------------------------------------------------------------------ slides
SLIDE_CSS = r"""
/* Layout: a lab-notebook page per slide, graph-paper ground, running header and footer. */
:root{
  --paper:#fbf7f0; --ink:#2b2219; --muted:#76685a; --rule:#e7ddcf; --grid:rgba(120,95,60,.06);
  --accent:#a65d0c; --teal:#2b5a5e; --tint:#f3eadb; --warn:#9b3d2e; --warntint:#f6e2dc; --codebg:#f2ebe0;
  --display:"Noto Serif SC","Songti SC","STSong",Georgia,serif;
  --body:"Noto Sans SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif;
  --mono:"JetBrains Mono","SFMono-Regular",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --paper:#1d1915; --ink:#eee5d8; --muted:#a9998a; --rule:#3b332b; --grid:rgba(230,200,160,.05);
  --accent:#e0a253; --teal:#86b9bb; --tint:#2a241e; --warn:#e48b78; --warntint:#33221d; --codebg:#26211b; color-scheme:dark}}
:root[data-theme="dark"]{
  --paper:#1d1915; --ink:#eee5d8; --muted:#a9998a; --rule:#3b332b; --grid:rgba(230,200,160,.05);
  --accent:#e0a253; --teal:#86b9bb; --tint:#2a241e; --warn:#e48b78; --warntint:#33221d; --codebg:#26211b; color-scheme:dark}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--body);line-height:1.65;
  background-image:linear-gradient(var(--grid) 1px,transparent 1px),linear-gradient(90deg,var(--grid) 1px,transparent 1px);
  background-size:24px 24px}
.bar{position:fixed;top:0;left:0;right:0;height:3px;background:var(--rule);z-index:5}
.bar i{display:block;height:100%;width:0;background:var(--accent);transition:width .2s}
.top{display:flex;gap:12px;align-items:center;justify-content:space-between;padding:12px 16px 0;font-size:13px;color:var(--muted)}
.top a{color:var(--muted)}
.top button{font:inherit;font-size:13px;color:var(--ink);background:transparent;border:1px solid var(--rule);border-radius:6px;padding:4px 10px;cursor:pointer}
.top button:focus-visible,.navbtn:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.deck{display:flex;flex-direction:column;gap:24px;padding:16px;align-items:center}
.slide{width:100%;max-width:1000px;background:var(--paper);border:1px solid var(--rule);border-radius:4px;
  padding:28px 28px 22px 44px;position:relative;display:flex;flex-direction:column;gap:14px;min-width:0}
.slide::before{content:"";position:absolute;left:28px;top:0;bottom:0;width:1px;background:var(--accent);opacity:.35}
.run{display:flex;justify-content:space-between;gap:12px;font-size:.78em;color:var(--muted);letter-spacing:.04em}
.run .day,.foot .pg,.pos{font-family:var(--mono)}
h1,h2{font-family:var(--display);font-weight:700;text-wrap:balance;margin:0;line-height:1.3}
h1{font-size:2.1em}
h2{font-size:1.5em}
.sub{color:var(--muted);font-size:1.05em;margin:0}
.body{display:flex;flex-direction:column;gap:12px;flex:1;min-width:0}
p{margin:0}
.lead{font-size:1.18em;max-width:40em}
ul.points,ol.steps{margin:0;padding-left:1.3em;display:flex;flex-direction:column;gap:6px}
ul.points li::marker{color:var(--accent)}
ol.steps li::marker{color:var(--accent);font-family:var(--mono);font-weight:600}
code{font-family:var(--mono);font-size:.9em;background:var(--codebg);padding:1px 5px;border-radius:3px}
strong{color:var(--accent);font-weight:600}
.formula{border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);padding:10px 4px;display:flex;flex-wrap:wrap;gap:6px 18px;align-items:baseline}
.fexpr{font-family:var(--mono);font-size:1.12em;color:var(--teal);font-weight:600;overflow-wrap:anywhere}
.fnote{color:var(--muted);font-size:.9em}
.note{background:var(--tint);padding:10px 14px;border-radius:4px;display:flex;gap:12px;align-items:baseline}
.note .tag{font-size:.75em;letter-spacing:.08em;color:var(--teal);font-weight:700;white-space:nowrap}
.note.warn{background:var(--warntint)}
.note.warn .tag{color:var(--warn)}
pre.code{background:var(--codebg);padding:12px 14px;border-radius:4px;overflow-x:auto;margin:0;font-size:.82em;line-height:1.5}
pre.code code{background:none;padding:0}
.tablewrap{overflow-x:auto}
table.t{border-collapse:collapse;font-size:.9em;font-variant-numeric:tabular-nums;min-width:60%}
table.t th{text-align:left;font-weight:600;border-bottom:1.5px solid var(--ink);padding:5px 12px 5px 0}
table.t td{border-bottom:1px solid var(--rule);padding:5px 12px 5px 0;vertical-align:top}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.cols>div{min-width:0;display:flex;flex-direction:column;gap:10px}
figure.plate{margin:0;background:#fdfaf5;border:1px solid var(--rule);border-radius:4px;padding:8px;max-width:min(100%,760px);align-self:flex-start}
figure.plate img{display:block;width:100%;height:auto}
figure.plate figcaption{color:#76685a;font-size:.8em;padding-top:4px}
.qa{border-left:2px solid var(--teal);padding-left:12px;display:flex;flex-direction:column;gap:4px}
.qa .q{font-weight:600}
.qa .a{color:var(--muted)}
.foot{display:flex;justify-content:space-between;font-size:.75em;color:var(--muted)}
.title-slide h1{font-size:2.6em}
.title-slide .kicker{color:var(--accent);font-size:.95em;letter-spacing:.06em}
.ctrl{display:flex;gap:8px;align-items:center}
.navbtn{display:none;min-width:34px}
.pos{display:none;font-size:12px;color:var(--muted);min-width:44px;text-align:center}
/* presentation mode: one slide at a time, scaled to the viewport */
body.present .deck{height:calc(100% - 48px);justify-content:center;padding:8px 16px 16px}
body.present .slide{display:none;width:var(--sw);max-width:none;aspect-ratio:16/9;font-size:calc(var(--sw) / 52);overflow:auto}
body.present .slide.on{display:flex}
body.present figure.plate img{max-height:calc(var(--sw) * 0.21);width:auto;max-width:100%}
body.present .formula{padding:6px 4px}
body.present .navbtn,body.present .pos{display:inline-block}
@media (max-width:760px){.cols{grid-template-columns:1fr}.slide{padding:22px 18px 18px 30px}.slide::before{left:16px}h1{font-size:1.7em}.title-slide h1{font-size:2em}}
@media (prefers-reduced-motion:reduce){.bar i{transition:none}}
@media print{body{background:none}.top,.bar{display:none}.deck{padding:0;gap:0}.slide{display:flex!important;page-break-after:always;border:none;aspect-ratio:auto;width:100%!important;font-size:14px!important}}
"""

SLIDE_JS = r"""
(function(){
  const slides=[...document.querySelectorAll('.slide')];
  const bar=document.querySelector('.bar i');
  const btn=document.getElementById('mode');
  const pos=document.getElementById('pos');
  let i=0;
  function fit(){
    const w=Math.min(window.innerWidth-32,(window.innerHeight-70)*16/9);
    document.documentElement.style.setProperty('--sw',Math.max(320,w)+'px');
  }
  function show(n){
    i=Math.max(0,Math.min(slides.length-1,n));
    slides.forEach((s,k)=>s.classList.toggle('on',k===i));
    bar.style.width=((i+1)/slides.length*100)+'%';
    pos.textContent=(i+1)+' / '+slides.length;
    if(document.body.classList.contains('present')){history.replaceState(null,'','#'+(i+1));}
  }
  function setMode(present){
    document.body.classList.toggle('present',present);
    btn.textContent=present?'全部展开':'逐页放映';
    try{localStorage.setItem('deckmode',present?'p':'s')}catch(e){}
    fit();show(i);
    if(!present&&slides[i])slides[i].scrollIntoView({block:'start'});
  }
  btn.addEventListener('click',()=>setMode(!document.body.classList.contains('present')));
  document.getElementById('prev').addEventListener('click',()=>show(i-1));
  document.getElementById('next').addEventListener('click',()=>show(i+1));
  document.addEventListener('keydown',e=>{
    if(!document.body.classList.contains('present'))return;
    if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(i+1)}
    if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(i-1)}
    if(e.key==='Home')show(0); if(e.key==='End')show(slides.length-1);
  });
  window.addEventListener('resize',fit);
  const h=parseInt((location.hash||'').slice(1),10); if(h>0)i=h-1;
  let pref=null; try{pref=localStorage.getItem('deckmode')}catch(e){}
  const wide=window.innerWidth>760;
  setMode(pref?pref==='p'&&wide:wide);
})();
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Noto+Sans+SC:wght@400;600&family=Noto+Serif+SC:wght@700&display=swap">')


def render_slides(lesson: Lesson, home_href="../../index.html") -> str:
    n = len(lesson.slides) + 1
    ltitle = cjk_punct(lesson.title)
    track = TRACK_LABEL[lesson.track]
    head = f"Day {lesson.day:02d}"
    parts = []
    goals = "".join(f"<li>{md_inline(g)}</li>" for g in lesson.goals)
    parts.append(f'''<section class="slide title-slide" aria-label="封面">
<div class="run"><span class="day">{head}</span><span>{esc(track)}</span></div>
<div class="body"><p class="kicker">{esc(COURSE_NAME)} · 第 {lesson.day} 天</p><h1>{esc(ltitle)}</h1>
<p class="sub">今天结束时你应该能够:</p><ul class="points">{goals}</ul></div>
<div class="foot"><span>建议用时 {esc(lesson.hours)}</span><span class="pg">1 / {n}</span></div></section>''')
    for k, (title, body) in enumerate(lesson.slides, start=2):
        title = cjk_punct(title)
        parts.append(f'''<section class="slide" aria-label="{esc(title)}">
<div class="run"><span><span class="day">{head}</span> · {esc(ltitle)}</span><span>{esc(track)}</span></div>
<h2>{esc(title)}</h2><div class="body">{body}</div>
<div class="foot"><span>{esc(COURSE_NAME)}</span><span class="pg">{k} / {n}</span></div></section>''')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Day {lesson.day:02d} {esc(ltitle)}</title>{FONTS}<style>{SLIDE_CSS}</style></head>
<body><div class="bar"><i></i></div>
<div class="top"><a href="{home_href}">课程首页</a><div class="ctrl"><button class="navbtn" id="prev" type="button" aria-label="上一页">‹</button><span class="pos" id="pos"></span><button class="navbtn" id="next" type="button" aria-label="下一页">›</button><button id="mode" type="button">逐页放映</button></div></div>
<main class="deck">{"".join(parts)}</main>
<script>{SLIDE_JS}</script></body></html>'''


# ------------------------------------------------------------------ notebook
SETUP = '''# 课程环境设置:让 notebook 找到课程代码库 src/course
import sys, pathlib
ROOT = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "src" / "course").exists())
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from course.plotting import use_style, OCHRE, TEAL, ROSE, SAND, INK
from course.check import check, check_true, summary
use_style()
pd.set_option("display.precision", 3)'''


def apply_solutions(code: str, lesson: Lesson) -> str:
    for name, expr in lesson.solutions.items():
        code = re.sub(rf"^(\s*){re.escape(name)}\s*=\s*None\b.*$", lambda m: f"{m.group(1)}{name} = {expr}", code, flags=re.M)
    for old, new in lesson.replacements:
        code = code.replace(old, new)
    return code


def render_notebook(lesson: Lesson, solutions: bool = False):
    head = (f"# Day {lesson.day:02d} · {cjk_punct(lesson.title)}\n\n"
            f"**{COURSE_NAME}** · {TRACK_LABEL[lesson.track]} · 建议用时 {lesson.hours}\n\n"
            "**今天的目标**\n\n" + "\n".join(f"- {g}" for g in lesson.goals) +
            "\n\n**怎么用这个 notebook**:先看同一文件夹里的 `slides.html`,再从上到下运行这里的单元格。"
            "标着 **练习** 的单元格需要你自己填写,填完运行下面的批改单元格,会显示 `正确`、`再想想` 或 `待完成`。"
            "所有数据都是课程编写或模拟的,没有真实来访者数据。")
    head = cjk_punct(head)
    cells = [new_markdown_cell(head), new_code_cell(SETUP)]
    if solutions:
        cells[0] = new_markdown_cell("**参考答案版**。请先独立完成 `lessons/` 里的练习,再来对照。\n\n" + head)
    for kind, text in lesson.cells:
        if kind == "md":
            cells.append(new_markdown_cell(cjk_punct(text.strip())))
        else:
            cells.append(new_code_cell(apply_solutions(text.strip(), lesson) if solutions else text.strip()))
    tail = []
    if lesson.interview:
        tail.append("## 口述题(录音,每题 90 秒)\n\n" + "\n".join(
            f"{i}. {q}\n   <details><summary>参考要点</summary>{a}</details>" for i, (q, a) in enumerate(lesson.interview, 1)))
    if lesson.research:
        tail.append("## 研究延伸\n\n" + lesson.research)
    for t in tail:
        cells.append(new_markdown_cell(cjk_punct(t)))
    cells.append(new_code_cell("summary()"))
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def render_sheet(lesson: Lesson) -> str:
    out = [f"# Day {lesson.day:02d} · {lesson.title}", "",
           f"{TRACK_LABEL[lesson.track]} · 建议用时 {lesson.hours}", "", "## 今天的目标", ""]
    out += [f"- {g}" for g in lesson.goals]
    out += ["", "## 学习顺序", "",
            "1. 看 `slides.html`(约 30 到 45 分钟)。",
            "2. 运行并完成 `notebook.ipynb` 里的练习(约 2 小时)。",
            "3. 精读今日论文,回答下面的问题(约 40 分钟)。",
            "4. 录音回答口述题(约 15 分钟)。", ""]
    if lesson.paper:
        p = lesson.paper
        out += ["## 今日论文", "", f"**{p.cite}**" + (f"  \n{p.link}" if p.link else ""), "", p.why, "", "阅读问题:", ""]
        out += [f"{i}. {q}" for i, q in enumerate(p.questions, 1)]
        out += [""]
    if lesson.homework:
        out += ["## 今日作业(提交到你的课程仓库)", ""] + [f"- [ ] {h}" for h in lesson.homework] + [""]
    if lesson.interview:
        out += ["## 口述题", ""] + [f"{i}. {q}" for i, (q, _) in enumerate(lesson.interview, 1)] + [""]
    if lesson.research:
        out += ["## 研究延伸", "", lesson.research, ""]
    return "\n".join(out)


def write_lesson(lesson: Lesson, out_root: Path = ROOT / "lessons"):
    d = out_root / lesson.folder
    d.mkdir(parents=True, exist_ok=True)
    (d / "slides.html").write_text(render_slides(lesson), encoding="utf-8")
    nbformat.write(render_notebook(lesson), d / "notebook.ipynb")
    (d / "README.md").write_text(cjk_punct(render_sheet(lesson)), encoding="utf-8")
    if lesson.solutions or lesson.replacements:
        sd = ROOT / "solutions"
        sd.mkdir(exist_ok=True)
        nbformat.write(render_notebook(lesson, solutions=True), sd / f"{lesson.folder}_solutions.ipynb")
    return d


# ------------------------------------------------------------------ index and document pages
WEEKS = {1: ("第 1 周", "测量基础:信度、一致性、区间、功效、IRT"),
         2: ("第 2 周", "题目、裁判与评分者:IRT 应用、偏好聚合、裁判偏差、污染"),
         3: ("第 3 周", "大模型、Agent 与项目实战"),
         4: ("第 4 周", "研究级评测与求职准备")}

PAGE_CSS = SLIDE_CSS + r"""
a{color:var(--teal)}
.page{max-width:980px;margin:0 auto;padding-block:28px 64px;display:flex;flex-direction:column;gap:28px}
body.doc{padding-inline:16px}
.hero{display:flex;flex-direction:column;gap:10px;padding-top:8px}
.hero h1{font-size:clamp(1.9rem,4vw,2.7rem)}
.hero .kicker{color:var(--accent);font-size:.9rem;letter-spacing:.06em}
.hero p{max-width:44em;color:var(--ink)}
.howto{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
.howto div{border-top:2px solid var(--ink);padding-top:8px;min-width:0}
.howto b{font-family:var(--mono);color:var(--accent)}
.week h2{font-size:1.35rem}
.week .wsub{color:var(--muted);margin:2px 0 10px}
.days{list-style:none;margin:0;padding:0;border-top:1px solid var(--rule)}
.days li{display:grid;grid-template-columns:3.2rem minmax(0,1fr) auto;gap:12px;align-items:center;padding:10px 0;border-bottom:1px solid var(--rule)}
.days .n{font-family:var(--mono);color:var(--muted);font-size:.9rem}
.days .t{min-width:0}
.days .t a{color:var(--ink);text-decoration:none;font-weight:600}
.days .t a:hover{color:var(--accent)}
.days .chip{display:inline-block;font-size:.72rem;color:var(--teal);border:1px solid var(--rule);border-radius:3px;padding:0 6px;margin-left:8px;white-space:nowrap}
.days .links{display:flex;gap:10px;font-size:.85rem;flex-wrap:wrap;justify-content:flex-end}
.days .links a{color:var(--muted)}
.days .links span{color:var(--muted)}
.days input{width:18px;height:18px;accent-color:var(--accent)}
.days li.done .t a{color:var(--muted);text-decoration:line-through}
.box{background:var(--tint);padding:16px 18px;border-radius:4px;display:flex;flex-direction:column;gap:8px}
.box h2{font-size:1.2rem}
.prog{font-family:var(--mono);color:var(--muted);font-size:.9rem}
.md h1{font-size:2rem;margin-top:8px}.md h2{font-size:1.35rem;margin-top:22px}.md h3{font-size:1.1rem;font-family:var(--body);margin-top:14px}
.md p,.md li{max-width:46em}.md table{border-collapse:collapse;font-size:.92rem}.md th,.md td{border-bottom:1px solid var(--rule);padding:5px 12px 5px 0;text-align:left;vertical-align:top}
.md pre{background:var(--codebg);padding:12px;border-radius:4px;overflow-x:auto}.md{display:flex;flex-direction:column;gap:10px}
.md ul,.md ol{display:flex;flex-direction:column;gap:4px}
@media (max-width:760px){.howto{grid-template-columns:1fr}.days li{grid-template-columns:2.6rem minmax(0,1fr)}.days .links{grid-column:2;justify-content:flex-start}}
"""

INDEX_JS = r"""
(function(){
  const boxes=[...document.querySelectorAll('.days input')];
  let state={}; try{state=JSON.parse(localStorage.getItem('course-done')||'{}')}catch(e){}
  function paint(){
    let n=0; boxes.forEach(b=>{const on=!!state[b.id]; b.checked=on; b.closest('li').classList.toggle('done',on); if(on)n++;});
    document.getElementById('prog').textContent='已完成 '+n+' / '+boxes.length+' 天';
  }
  boxes.forEach(b=>b.addEventListener('change',()=>{state[b.id]=b.checked; try{localStorage.setItem('course-done',JSON.stringify(state))}catch(e){} paint();}));
  paint();
})();
"""


def render_index(lessons, site: bool = False) -> str:
    weeks = {}
    for l in lessons:
        weeks.setdefault((l.day - 1) // 7 + 1, []).append(l)
    blocks = []
    for w, ls in sorted(weeks.items()):
        name, sub = WEEKS[w]
        items = []
        for l in ls:
            base = f"lessons/{l.folder}"
            nb = (f'<span title="在课程包里打开">notebook 在课程包</span>' if site else f'<a href="{base}/notebook.ipynb">notebook</a>')
            sheet = "" if site else f'<a href="{base}/README.md">学习单</a>'
            items.append(f'''<li><span class="n">D{l.day:02d}</span><span class="t"><a href="{base}/slides.html">{esc(cjk_punct(l.title))}</a><span class="chip">{esc(TRACK_LABEL[l.track])}</span></span>
<span class="links"><a href="{base}/slides.html">幻灯片</a>{nb}{sheet}<input type="checkbox" id="done-{l.day}" aria-label="第 {l.day} 天已完成"></span></li>''')
        blocks.append(f'<section class="week"><h2>{esc(name)}</h2><p class="wsub">{esc(cjk_punct(sub))}</p><ul class="days">{"".join(items)}</ul></section>')
    head = "" if site else ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
                            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">')
    tail = "" if site else "</body></html>"
    open_body = "" if site else "</head><body class=\"doc\">"
    cw = '<a href="coursework.html">期末作业说明</a>'
    return f'''{head}<title>心理大模型评测课</title>{FONTS}<style>{PAGE_CSS}{"body{padding-inline:16px}" if site else ""}</style>{open_body}
<main class="page">
<header class="hero"><p class="kicker">28 天 · 每天幻灯片 + notebook + 论文精读</p><h1>{esc(COURSE_NAME)}</h1>
<p>{cjk_punct("面向心理学背景的研究者:用心理测量和实验设计的方法评测心理健康大模型,并动手搭建一个有评测的 CBT 支持 Agent。所有例子都来自心理健康场景,所有数据都是课程编写或模拟的。")}</p>
<p class="prog" id="prog"></p></header>
<section class="howto" aria-label="怎么学">
<div><b>1</b> {cjk_punct("看当天的幻灯片,点“逐页放映”可以全屏翻页。")}</div>
<div><b>2</b> {cjk_punct("在 Jupyter 里运行当天的 notebook,完成练习,看自动批改结果。")}</div>
<div><b>3</b> {cjk_punct("读当天的论文,录音回答口述题,把作业推到你的仓库。")}</div></section>
{"".join(blocks)}
<section class="box"><h2>期末作业</h2><p>{cjk_punct("一项可以延伸成你自己研究的小型研究:检验 LLM 裁判评价 CBT 取向回复的效度,并评测一个你自己搭建的支持 Agent。第 28 天提交。")}</p><p>{cw}</p></section>
</main><script>{INDEX_JS}</script>{tail}'''


def render_doc_page(md_text: str, title: str, site: bool = False, home="index.html") -> str:
    import markdown as mdlib
    body = mdlib.markdown(cjk_punct(md_text), extensions=["tables", "fenced_code"])
    head = "" if site else ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
                            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">')
    open_body = "" if site else "</head><body class=\"doc\">"
    tail = "" if site else "</body></html>"
    return f'''{head}<title>{esc(title)}</title>{FONTS}<style>{PAGE_CSS}</style>{open_body}
<main class="page md"><p><a href="{home}">课程首页</a></p>{body}</main>{tail}'''
