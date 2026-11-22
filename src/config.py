"""Paths and constants shared by the whole project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_ZIP = RAW_DIR / "incident.zip"
RAW_CSV = RAW_DIR / "incident_event_log.csv"
DB_PATH = DATA_DIR / "tickets.duckdb"

SQL_DIR = ROOT / "sql"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/498/"
    "incident+management+process+enriched+event+log.zip"
)

# SQL files are run in this order to build the database.
SQL_FILES = ["01_events.sql", "02_incidents.sql", "03_snapshot.sql"]
