"""Run the whole chain from raw data to tables, figures and the demo's data: python run_all.py

Most of the time goes to fitting the survival forest: once for the main result (src/evaluate.py)
and again for each check that refits it (src/checks.py).
"""

import warnings

from src import checks, db, evaluate
from src.analysis import mean_trap, model_report, waiting


def main() -> None:
    warnings.filterwarnings("ignore")
    db.build()
    mean_trap.main()
    waiting.main()
    checks.main(evaluate.main())
    model_report.main()


if __name__ == "__main__":
    main()
