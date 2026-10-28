"""Export a tagged Git snapshot plus explicitly selected reference documents."""

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import subprocess
import tomllib
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "EdgeInferenceLab/"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--references", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=ROOT, check=True)
    version = tomllib.loads(git("show", "HEAD:pyproject.toml").decode("utf-8"))["project"]["version"]
    tag = git("describe", "--tags", "--exact-match", "HEAD").decode().strip()
    assert tag == f"v{version}", "The release tag must match the committed project version."
    references = [path.resolve(strict=True) for path in args.references]
    assert len({p.name for p in references}) == len(references), "Reference filenames must be distinct."
    assert all(p.is_file() for p in references)
    output = args.out.resolve()
    assert not output.exists(), "Choose a new output filename for each exported discussion bundle."
    archive = BytesIO(git("archive", "--format=zip", f"--prefix={PREFIX}", "HEAD"))
    with ZipFile(archive, "a", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        tracked_count = len(bundle.namelist())
        for path in references:
            bundle.write(path, PREFIX + "references/" + path.name)
        info = {
            "project": "EdgeInferenceLab", "version": version, "tag": tag,
            "commit": git("rev-parse", "HEAD").decode().strip(),
            "repository": git("remote", "get-url", "origin").decode().strip(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "committed Git snapshot", "git_archive_entries": tracked_count,
            "reference_files": ["references/" + p.name for p in references],
            "start_here": "CHAT_CONTEXT.md",
        }
        bundle.writestr(PREFIX + "BUNDLE_INFO.json", json.dumps(info, ensure_ascii=False, indent=2) + "\n")
        contents = sorted(bundle.namelist() + [PREFIX + "BUNDLE_CONTENTS.txt"])
        bundle.writestr(PREFIX + "BUNDLE_CONTENTS.txt", "\n".join(contents) + "\n")
        assert bundle.testzip() is None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(archive.getvalue())
    print(json.dumps({"path": str(output), "bytes": output.stat().st_size,
                      "entries": len(contents), "version": version, "tag": tag}, ensure_ascii=False))


if __name__ == "__main__":
    main()
