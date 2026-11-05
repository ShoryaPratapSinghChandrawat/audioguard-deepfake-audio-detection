"""Inspect real files before parsing official LA protocols; reject ambiguous layouts.

No file names or partition folders are guessed. Users map inspected paths in YAML.
The supported five-column format is validated, not silently assumed.
"""
from collections import Counter
from itertools import combinations
from pathlib import Path
from src.common import digest

PARTITIONS = ("train", "dev", "eval")


def inspect(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}. Supply the official LA data first.")
    counts, candidates = Counter(), []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        counts[path.suffix.lower()] += 1
        if path.suffix.lower() in (".txt", ".trl", ".trn"):
            with path.open(encoding="utf-8", errors="replace") as handle:
                preview = [line.strip() for _, line in zip(range(3), handle)]
            candidates.append({"path": str(path), "preview": preview,
                               "preview_field_counts": [len(s.split()) for s in preview]})
    result = {"root": str(root.resolve()), "files_by_extension": dict(counts),
              "protocol_candidates": candidates,
              "instruction": "Map actual train/dev/eval protocol and audio paths in YAML; do not random-split."}
    print(result)
    return result


def parse_protocol(path, partition):
    rows, ids = [], set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{path}:{line_number}: expected validated LA five-column protocol, got {len(fields)} fields")
            speaker, utterance, placeholder, attack, label = fields
            if label not in ("bonafide", "spoof") or placeholder != "-":
                raise ValueError(f"{path}:{line_number}: unsupported protocol: {fields}")
            if (label == "bonafide" and attack != "-") or (label == "spoof" and not attack.startswith("A")):
                raise ValueError(f"Inconsistent attack/label: {fields}")
            if utterance in ids:
                raise ValueError(f"Duplicate trial {utterance}")
            ids.add(utterance)
            rows.append({"speaker": speaker, "utterance": utterance, "attack": attack,
                         "label": int(label == "spoof"), "partition": partition})
    if not rows or {r["label"] for r in rows} != {0, 1}:
        raise ValueError(f"{path}: both classes required")
    return rows


def verify_disjoint(partitions):
    checks = {}
    for a, b in combinations(PARTITIONS, 2):
        for field in ("speaker", "utterance"):
            overlap = sorted({r[field] for r in partitions[a]} & {r[field] for r in partitions[b]})
            checks[f"{a}_{b}_{field}_overlap"] = overlap
    print("Partition disjointness:", checks)
    if any(checks.values()):
        raise ValueError(f"Partition leakage detected: {checks}")
    return checks


def counts(rows):
    return {"total": len(rows), "bonafide": sum(r["label"] == 0 for r in rows),
            "spoof": sum(r["label"] == 1 for r in rows),
            "speakers": len({r["speaker"] for r in rows}),
            "attacks": dict(Counter(r["attack"] for r in rows))}


def load_dataset(cfg):
    ds = cfg["dataset"]
    # Inspection is deliberately first, before protocol parsing or audio assumptions.
    inspection = inspect(ds["root"])
    partitions, hashes = {}, {}
    for part in PARTITIONS:
        protocol, audio = ds["protocols"][part], ds["audio_roots"][part]
        if not protocol or not audio:
            raise ValueError("Inspect dataset, then set all protocols and audio_roots in YAML.")
        if not Path(audio).is_dir():
            raise FileNotFoundError(audio)
        partitions[part] = parse_protocol(protocol, part)
        hashes[part] = digest(protocol)
    checks = verify_disjoint(partitions)
    full_counts = {p: counts(rows) for p, rows in partitions.items()}
    limit = ds["max_samples_per_partition"]
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        raise ValueError("Subset limit must be a positive integer or null")
    for part in PARTITIONS:
        # Sorted-prefix subset is explicit and reproducible, not a new random split.
        rows = sorted(partitions[part], key=lambda r: r["utterance"])
        partitions[part] = rows if limit is None else rows[:limit]
        if {r["label"] for r in partitions[part]} != {0, 1}:
            raise ValueError(f"{part} subset lacks one class; increase limit or use all trials")
        index = {}
        for path in sorted(Path(ds["audio_roots"][part]).rglob("*")):
            if path.suffix.lower() in (".flac", ".wav"):
                if path.stem in index:
                    raise ValueError(f"Ambiguous audio stem: {path.stem}")
                index[path.stem] = str(path)
        for row in partitions[part]:
            if row["utterance"] not in index:
                raise FileNotFoundError(f"Missing audio for {part}/{row['utterance']}")
            row["path"] = index[row["utterance"]]
    audit = {"inspection": inspection, "checks": checks, "protocol_sha256": hashes,
             "full_counts": full_counts, "used_counts": {p: counts(r) for p, r in partitions.items()},
             "subset_requested": limit is not None,
             "subset_selection": "sorted utterance prefix" if limit is not None else "all protocol trials"}
    print("Dataset audit:", audit)
    return partitions, audit
