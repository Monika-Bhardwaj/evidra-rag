from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

from src.config import get_settings
from src.evaluation.evaluator import Evaluator


def main() -> None:
    evaluator = Evaluator(get_settings())
    result = evaluator.run()
    print(json.dumps(result["summary"], indent=2))
    report = get_settings().eval_report_path_resolved
    print(f"\nReport written to {report}")
    print("Plots written to docs/eval_plots/")


if __name__ == "__main__":
    main()