"""Run the whole chain from raw data to tables, figures and the demo's data: python run_all.py

Takes about 7 minutes; almost all of it is fitting the survival forest (src/evaluate.py).
"""

import warnings

from src import db, evaluate
from src.analysis import mean_trap, model_report, waiting


def main() -> None:
    warnings.filterwarnings("ignore")
    db.build()
    mean_trap.main()
    waiting.main()
    evaluate.main()
    model_report.main()


if __name__ == "__main__":
    main()
