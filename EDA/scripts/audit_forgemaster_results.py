"""Read back saved outputs and audit timestamps independently of EDA queries."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

EDA=Path(__file__).resolve().parents[1]
OUT=EDA / "data/processed/forgemaster_v1"
METRICS=EDA / "reports/forgemaster_eda_metrics.json"


def main():
    metrics=json.loads(METRICS.read_text("utf-8"))
    con=duckdb.connect(str(OUT / "eda.duckdb"),read_only=True)
    con.execute("SET threads=2")
    published=int(datetime.fromisoformat(metrics["source_manifest"]["source_last_updated"].replace("Z","+00:00")).timestamp())
    current=int(datetime(2026,11,9,tzinfo=timezone.utc).timestamp())
    values=con.execute("""SELECT
        count(*) FILTER(WHERE unix_timestamp_updated>?) updated_after_publication_rows,
        count(*) FILTER(WHERE unix_timestamp_updated>=?) updated_after_analysis_date_rows,
        count(*) FILTER(WHERE unix_timestamp_updated<0) negative_updated_timestamp_rows,
        min(unix_timestamp_created),max(unix_timestamp_created),
        min(unix_timestamp_updated),max(unix_timestamp_updated) FROM typed""",[published,current]).fetchone()
    keys=["updated_after_publication_rows","updated_after_analysis_date_rows","negative_updated_timestamp_rows",
          "first_created","last_created","first_updated","last_updated"]
    audit=dict(zip(keys,values))
    for key in keys[3:]:
        audit[key+"_utc"]=datetime.fromtimestamp(audit.pop(key),tz=timezone.utc).isoformat()
    clean=pq.ParquetFile(OUT / "clean_pairs.parquet")
    accounts=pq.ParquetFile(OUT / "potential_test_accounts.parquet")
    checks={"clean_parquet_rows":clean.metadata.num_rows==metrics["clean"]["rows"],
        "account_parquet_rows":accounts.metadata.num_rows==metrics["eligibility"][2]["eligible_users"],
        "clean_steamids_are_strings":str(clean.schema_arrow.field("steamid").type)=="string",
        "account_steamids_are_strings":str(accounts.schema_arrow.field("steamid").type)=="string"}
    if not all(checks.values()):
        raise AssertionError(checks)
    frank_path=EDA / "data/processed/forgemaster_eda/frank_reviews_identity.parquet"
    frank_counts=con.execute("""WITH counts AS (SELECT appid,count(*) n FROM read_parquet(?) GROUP BY appid)
        SELECT count(*) games,min(n) minimum,max(n) maximum,
        count(*) FILTER(WHERE n=100) games_exact_100,
        count(*) FILTER(WHERE n>100) games_gt_100 FROM counts""",[str(frank_path)]).fetchone()
    metrics['frank_comparison']['raw_per_game_counts']=dict(zip(
        ['games','minimum','maximum','games_exact_100','games_gt_100'],frank_counts))
    metrics["additional_saved_output_audit"]={"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timestamps":audit,"checks":checks,
        "publication_cutoff_utc":datetime.fromtimestamp(published,tz=timezone.utc).isoformat(),
        "analysis_date_end_exclusive_utc":"2026-11-09T00:00:00+00:00"}
    METRICS.write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    con.close()
    print(json.dumps(metrics["additional_saved_output_audit"],indent=2))


if __name__=="__main__":
    main()
