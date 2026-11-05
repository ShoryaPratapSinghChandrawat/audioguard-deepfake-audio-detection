"""Create a portable source ZIP with checksums, excluding data, weights and caches.

This packages real files (never a patch). Rerun after any source change.
"""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    root = Path(__file__).resolve().parents[1]
    target = root.parent / "AudioGuard-source.zip"
    excluded_roots = {"data", "models", "results", ".venv"}
    excluded_anywhere = {"__pycache__", ".pytest_cache", ".git"}
    # Exclude root artifact folders, NOT src/data or src/models source packages.
    files = [p for p in sorted(root.rglob("*")) if p.is_file()
             and p.relative_to(root).parts[0] not in excluded_roots
             and not any(part in excluded_anywhere for part in p.relative_to(root).parts)
             and p.name != "MANIFEST.sha256.json" and p.suffix != ".zip"]
    # Preserve empty required directories through their small placeholder files.
    files += [root / d / ".gitkeep" for d in ("data", "models", "results")]
    manifest = {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (root / "MANIFEST.sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in files + [root / "MANIFEST.sha256.json"]:
            archive.write(path, "AudioGuard/" + path.relative_to(root).as_posix())
    print(target)


if __name__ == "__main__":
    main()
