"""Independently count every source record with Python's CSV parser.

Collect RFC-strict failures separately; the full scan uses csv's usual
permissive quote handling and fails if any parsed record has !=13 columns.
"""
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EDA=Path(__file__).resolve().parents[1]
RAW=EDA / "data/raw/forgemaster_v1"
OUT=EDA / "data/processed/forgemaster_v1/python_csv_validation.json"
csv.field_size_limit(16777216)


def scan(path):
    strict_error=None
    try:
        with path.open(encoding="utf-8",newline="") as stream:
            reader=csv.reader(stream,strict=True)
            for row in reader:
                pass
    except csv.Error as error:
        strict_error={"physical_line":reader.line_num,"error":str(error)}
    rows=0
    with path.open(encoding="utf-8",newline="") as stream:
        reader=csv.reader(stream)
        header=next(reader)
        for row in reader:
            rows+=1
            if len(row)!=13:
                raise ValueError(f"{path.name}: record {rows}: {len(row)} columns, physical line {reader.line_num}")
    result={"file":path.name,"rows":rows,"columns":len(header),"strict_csv_error":strict_error}
    print(json.dumps(result),flush=True)
    return result


def main():
    paths=sorted(RAW.glob("reviews-*.csv"))
    if len(paths)!=11:
        raise ValueError(f"Expected 11 files, got {len(paths)}")
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(scan,paths))
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"method":"Python csv.reader, UTF-8 strict decoding; full 13-column scan, zero skipped records",
        "rows":sum(row["rows"] for row in results),"files":results},indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
