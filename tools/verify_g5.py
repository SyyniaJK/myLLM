"""Rebuild G5 controls/frontiers, replay saved scan cells and regenerate summaries."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from edgeinfer.diagnostics import composition_controls, length_controls, replay_g5
from edgeinfer.experiment import write_json
from edgeinfer.gap_analysis import analyze_g5
from edgeinfer.snapshot import exact_controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    subprocess.run(["git", "diff", "--exit-code", "v0.1.0", "--", "results/g123_final", "results/g4", "results/g4_training"], check=True)
    replay_count = 0
    for replicate in plan["replicates"]:
        for method in plan["methods"]:
            replay_g5(directory, "think_0.5_atten_40", method, "equal_uplink", replicate, plan["seeds"][1],
                      directory / "replay_checks" / f"stress_{method}_{replicate}")
            replay_count += 1
    for method in plan["methods"]:
        replay_g5(directory, "think_2_atten_0", method, "tp", 1, plan["seeds"][0],
                  directory / "replay_checks" / f"nominal_{method}")
        replay_count += 1
    with tempfile.TemporaryDirectory(prefix="edgeinfer-g5-verify-") as temporary:
        repeat = Path(temporary).resolve()
        assert repeat.parent == Path(tempfile.gettempdir()).resolve()
        shutil.copytree(directory / "inputs", repeat / "inputs")
        composition_controls(plan, repeat)
        length_controls(plan, repeat)
        exact_controls(plan, repeat)
        rebuilt = ["composition.csv", "length.csv", "exact.csv", "exact_frontier.csv"]
        assert all((directory / name).read_bytes() == (repeat / name).read_bytes() for name in rebuilt)
    summaries = [f"{name}.{ext}" for name in ("scan_summary", "composition_summary", "length_summary", "exact_summary", "wireless_differences")
                 for ext in ("csv", "json")] + ["report.md"]
    before = {name: (directory / name).read_bytes() for name in summaries}
    analyze_g5(directory)
    assert all((directory / name).read_bytes() == data for name, data in before.items())
    verification = {"exact_scan_replays": replay_count, "regenerated_controls_exact": rebuilt,
        "regenerated_summaries_exact": summaries, "versioned_g1_g4_results_unchanged": True,
        "new_training_runs": 0, "numerical_code_commit": plan["code_commit"]}
    write_json(directory / "verification.json", verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
