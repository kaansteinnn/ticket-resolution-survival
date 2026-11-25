"""Build and query the DuckDB database.

Run `python -m src.db` to download the data (if missing) and rebuild `data/tickets.duckdb`.
"""

from __future__ import annotations

import urllib.request
import zipfile

import duckdb
import pandas as pd

from src import config


def download_raw() -> None:
    """Download and unzip the UCI event log unless the CSV is already there."""
    if config.RAW_CSV.exists():
        return
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {config.DATA_URL}")
    urllib.request.urlretrieve(config.DATA_URL, config.RAW_ZIP)
    with zipfile.ZipFile(config.RAW_ZIP) as zf:
        zf.extract(config.RAW_CSV.name, config.RAW_DIR)


def run_sql_files(con: duckdb.DuckDBPyConnection, files: list[str] | None = None) -> None:
    for name in files or config.SQL_FILES:
        con.execute((config.SQL_DIR / name).read_text(encoding="utf-8"))


def build() -> None:
    """Rebuild the database from the raw CSV by running the files in sql/ in order."""
    download_raw()
    config.DB_PATH.unlink(missing_ok=True)
    con = duckdb.connect(str(config.DB_PATH))
    csv_path = str(config.RAW_CSV).replace("'", "''")
    con.execute(f"set variable csv_path = '{csv_path}'")
    run_sql_files(con)
    n_events, n_incidents = con.execute(
        "select (select count(*) from events), (select count(*) from incidents)"
    ).fetchall()[0]
    con.close()
    print(f"Built {config.DB_PATH.name}: {n_events:,} events, {n_incidents:,} incidents")


def connect() -> duckdb.DuckDBPyConnection:
    """Read-only connection to the built database."""
    if not config.DB_PATH.exists():
        raise FileNotFoundError(f"{config.DB_PATH} not found. Run `python -m src.db` first.")
    return duckdb.connect(str(config.DB_PATH), read_only=True)


def snapshot(con: duckdb.DuckDBPyConnection, cutoff, only_open: bool = False) -> pd.DataFrame:
    """Every ticket visible at `cutoff`, exactly as the system showed it then."""
    macro = "open_at" if only_open else "snapshot_at"
    ts = pd.Timestamp(cutoff).strftime("%Y-%m-%d %H:%M:%S")
    return con.execute(f"select * from {macro}(timestamp '{ts}') order by number").df()


def incidents(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Final outcome of every ticket (hindsight). Never used as model input."""
    return con.execute("select * from incidents order by number").df()


def never_handled_by(con: duckdb.DuckDBPyConnection, team: str) -> set[str]:
    """Tickets that were not assigned to `team` at any point of their life (hindsight; descriptive use only)."""
    rows = con.execute(
        "select number from incidents where number not in (select number from events where assignment_group = ?)",
        [team],
    ).fetchall()
    return {number for (number,) in rows}


if __name__ == "__main__":
    build()
