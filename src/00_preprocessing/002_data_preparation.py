from pathlib import Path
import duckdb
import os
import psutil

# ---------- PATHS ----------

BASE_DIR = Path.cwd()
PROJECT_ROOT = BASE_DIR.parents[1]

INPUT_PATTERN = (
    PROJECT_ROOT / "results" / "00_preprocessing" / "daily"/ "user_daily_activity_*.parquet"
).as_posix()

PREDICTION_OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "00_preprocessing" / "user_summary" / "pred.parquet"
).as_posix()

CATEGORIZATION_OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "00_preprocessing" /  "user_summary" / "cat.parquet"
).as_posix()

PREDICTION_SQL_FILE = PROJECT_ROOT / "sql" / "00_prediction_user_summary_180d.sql"
CATEGORIZATION_SQL_FILE = PROJECT_ROOT / "sql" / "00_categorization_user_summary.sql"


# ---------- DuckDB-Config ----------

total_ram_gb = psutil.virtual_memory().total / 1024**3
duckdb_ram_gb = max(1, int(total_ram_gb * 0.6))  # 60 % of total RAM
num_threads = max(1, (os.cpu_count() or 4) - 2)   # 2 cores less than total

# ---------- Helper ----------

def run_sql_file(con: duckdb.DuckDBPyConnection, sql_file: Path, output_path: str) -> None:
    sql_template = sql_file.read_text()
    sql = sql_template.format(
        input_pattern=INPUT_PATTERN,
        output_path=output_path,
    )
    con.sql(sql)


def main() -> None:
    con = duckdb.connect(
        config={"threads": num_threads, "max_memory": f"{duckdb_ram_gb}GB"}
    )

    # both pipelines
    run_sql_file(con, PREDICTION_SQL_FILE, PREDICTION_OUTPUT_PATH)
    run_sql_file(con, CATEGORIZATION_SQL_FILE, CATEGORIZATION_OUTPUT_PATH)


# ---------- SCRIPT ----------

if __name__ == "__main__":
    print("Base dir        :", BASE_DIR)
    print("PROJECT_ROOT    :", PROJECT_ROOT)
    print("DuckDB threads  :", num_threads)
    print("DuckDB max_mem  :", f"{duckdb_ram_gb}GB")
    main()

# ---------- END OF SCRIPT ----------