from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.orchestrator import run_case


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full layout pipeline and store outputs per case")
    parser.add_argument("--input", required=True, help="Path to UI input JSON")
    parser.add_argument("--user", required=True, help="user_id")
    parser.add_argument("--description", default="", help="user description")
    parser.add_argument("--notes", default="", help="user special notes")
    parser.add_argument("--cases-root", default="cases", help="root folder for cases")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    payload = json.loads(input_path.read_text())

    result = run_case(
        input_payload=payload,
        user_id=args.user,
        description=args.description or None,
        special_notes=args.notes or None,
        cases_root=args.cases_root,
    )

    print(json.dumps({"case_id": result["case_id"], "case_dir": result["case_dir"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
