"""메인 Kieran → frank EDA 노트북 변환 후 실행.

사용법 (EDA/ 폴더 기준):
    python scripts/build_notebook.py            # 변환 + 실행(출력 포함 저장)
    python scripts/build_notebook.py --no-run   # 변환만
    python scripts/build_notebook.py 01_kieran_frank_eda  # 지정 노트북

셀 구분: '# %%' 는 코드 셀, '# %% [markdown]' 는 마크다운 셀(각 줄의 '# ' 접두어 제거).
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat

EDA_DIR = Path(__file__).resolve().parent.parent


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
    ap.add_argument("notebook", nargs="?", default="01_kieran_frank_eda", help="notebooks/ 아래의 노트북 이름(.py/.ipynb 생략 가능)")
    ap.add_argument("--no-run", action="store_true")
    args = ap.parse_args()
    stem = Path(args.notebook).stem
    src = EDA_DIR / "notebooks" / f"{stem}.py"
    dst = EDA_DIR / "notebooks" / f"{stem}.ipynb"
    nb = nbformat.v4.new_notebook()
    nb.cells = parse(src.read_text("utf-8"))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    # Keep the last successful notebook if conversion or execution fails.
    # The staging file stays beside dst so relative paths and kernel cwd agree.
    with tempfile.NamedTemporaryFile(prefix=f".{stem}-", suffix=".ipynb", dir=dst.parent, delete=False) as f:
        staged = Path(f.name)
    try:
        nbformat.write(nb, staged)
        if not args.no_run:
            subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                            "--ExecutePreprocessor.timeout=3600", str(staged)], check=True, cwd=dst.parent)
        staged.replace(dst)
        print("wrote", dst)
    finally:
        staged.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
