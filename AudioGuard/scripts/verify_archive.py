"""Execute an extracted source ZIP to catch missing files, not just bad hashes.

Synthetic ablations run in a temporary clean workspace, with no access to the
local source tree. All reported status is written by this rerunnable script.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="../AudioGuard-source.zip")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        with ZipFile(Path(args.zip).resolve()) as archive:
            for name in archive.namelist():
                if not name.startswith("AudioGuard/") or ".." in Path(name).parts:
                    raise ValueError("Unexpected archive member: " + name)
            manifest = json.loads(archive.read("AudioGuard/MANIFEST.sha256.json"))
            for name, expected in manifest.items():
                if hashlib.sha256(archive.read("AudioGuard/" + name)).hexdigest() != expected:
                    raise ValueError("Archive checksum mismatch: " + name)
            archive.extractall(temporary)
        result = subprocess.run([sys.executable, "scripts/research_smoke.py"],
                                cwd=Path(temporary) / "AudioGuard", capture_output=True, text=True,
                                env=dict(os.environ, GRADIO_ANALYTICS_ENABLED="False"))
        write_json("docs/archive_validation.json", {"scope": "EXTRACTED ZIP: SYNTHETIC SOFTWARE TESTS ONLY",
                   "manifest_verified": True, "exit_code": result.returncode,
                   "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise SystemExit("Extracted ZIP smoke failed; see docs/archive_validation.json")
    print("Archive hashes and fresh-extraction research smoke passed.")


if __name__ == "__main__":
    main()
