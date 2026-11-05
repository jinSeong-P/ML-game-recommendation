"""notebooks/01_eda.py (percent 형식) → notebooks/01_eda.ipynb 변환 후 실행.

사용법 (EDA/ 폴더 기준):
    python scripts/build_notebook.py            # 변환 + 실행(출력 포함 저장)
    python scripts/build_notebook.py --no-run   # 변환만

셀 구분: '# %%' 는 코드 셀, '# %% [markdown]' 는 마크다운 셀(각 줄의 '# ' 접두어 제거).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import nbformat

EDA_DIR = Path(__file__).resolve().parent.parent
SRC = EDA_DIR / "notebooks" / "01_eda.py"
DST = EDA_DIR / "notebooks" / "01_eda.ipynb"


def parse(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "markdown":
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(body))
        elif body:
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true")
    args = ap.parse_args()
    nb = nbformat.v4.new_notebook()
    nb.cells = parse(SRC.read_text("utf-8"))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbformat.write(nb, DST)
    print("wrote", DST)
    if not args.no_run:
        subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                        "--ExecutePreprocessor.timeout=3600", str(DST)], check=True, cwd=DST.parent)


if __name__ == "__main__":
    main()
