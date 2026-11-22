"""Run the whole chain from raw data to tables and figures: python run_all.py"""

import warnings

from src import db
from src.analysis import mean_trap, waiting


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    db.build()
    mean_trap.main()
    waiting.main()


if __name__ == "__main__":
    main()
