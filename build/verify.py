"""Execute notebooks; for solution notebooks, require every check to pass."""
import re
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
args = [a for a in sys.argv[1:] if not a.startswith("--")]
pattern = args[0] if args else "day"
no_write = "--no-write" in sys.argv
bad = []
for nb_path in sorted(list((ROOT / "lessons").glob(f"{pattern}*/notebook.ipynb")) + sorted((ROOT / "solutions").glob(f"{pattern}*_solutions.ipynb"))):
    nb = nbformat.read(nb_path, 4)
    try:
        ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})
    except Exception as e:
        bad.append((nb_path.name, str(e)[-400:]))
        print("ERROR", nb_path.relative_to(ROOT), str(e)[-300:])
        continue
    out = "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "stream")
    if "solutions" in nb_path.name:
        fails = re.findall(r"\[ (再想想|待完成) \][^\n]*", out)
        status = "all checks pass" if not fails else f"FAILED: {fails}"
        if fails:
            bad.append((nb_path.name, fails))
    else:
        status = "ran"
        if not no_write:
            nbformat.write(nb, nb_path)   # keep outputs so the reader sees figures
    print(f"{nb_path.relative_to(ROOT)}: {status}")
print("problems:", len(bad))
sys.exit(1 if bad else 0)
