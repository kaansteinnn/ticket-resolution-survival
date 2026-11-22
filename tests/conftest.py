import duckdb
import pytest

from src import config, db


@pytest.fixture(scope="session")
def con():
    """Read-only connection to the built database; tests are skipped if it has not been built."""
    if not config.DB_PATH.exists():
        pytest.skip("database not built; run `python -m src.db`")
    connection = db.connect()
    yield connection
    connection.close()


@pytest.fixture()
def scratch_con():
    """In-memory database holding a private, writable copy of `events` plus the snapshot macros."""
    if not config.DB_PATH.exists():
        pytest.skip("database not built; run `python -m src.db`")
    connection = duckdb.connect()
    connection.execute(f"attach '{config.DB_PATH}' as built (read_only)")
    connection.execute("create table events as select * from built.events")
    connection.execute("detach built")
    db.run_sql_files(connection, ["03_snapshot.sql"])
    yield connection
    connection.close()
