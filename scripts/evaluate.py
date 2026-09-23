from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.evaluation.evaluator import Evaluator


def archive_run(settings) -> Path:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_dir = settings.root_dir / "docs" / "eval_results" / run_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        settings.processed_dir_path / "evaluation_results.json",
        archive_dir / "evaluation_results.json",
    )
    claims_path = settings.processed_dir_path / "claims.jsonl"
    if claims_path.exists():
        shutil.copy2(claims_path, archive_dir / "claims.jsonl")
    shutil.copy2(settings.eval_report_path_resolved, archive_dir / "evaluation_report.md")
    abstain_path = settings.processed_dir_path / "abstention_results.json"
    if abstain_path.exists():
        shutil.copy2(abstain_path, archive_dir / "abstention_results.json")
    return archive_dir


def main() -> None:
    settings = get_settings()
    evaluator = Evaluator(settings)
    result = evaluator.run()
    print(json.dumps(result["summary"], indent=2))
    archive_dir = archive_run(settings)
    print(f"\nReport written to {settings.eval_report_path_resolved}")
    print("Plots written to docs/eval_plots/")
    print(f"Run archived to {archive_dir}")


if __name__ == "__main__":
    main()
