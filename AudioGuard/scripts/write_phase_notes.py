"""Write phase notes after smoke tests, linking all test claims to saved JSON evidence.

The notes intentionally contain no benchmark performance values. The synthetic
validation JSON is an exact copy of the smoke runner's script-written status.
"""
import json
from pathlib import Path

PHASES = {
    1: {
        "title": "Workspace and official-data audit",
        "built": "Isolated workspace, pinned dependencies, explicit YAML paths, real-file inspector, validated protocol parser, and full-protocol speaker/utterance-disjointness checks.",
        "why": "The existing repository is unrelated and must remain intact. Inspection prevents guessed paths and formats. Official partitions preserve the challenge generalization setting. Hashes and exact counts reveal data drift and subsets.",
        "learn": "Read the protocol parser and inspect your real protocol lines. Explain speaker leakage versus utterance leakage, metadata-only auditing versus model fitting, and why a sorted-prefix debugging subset is not representative.",
        "stages": ["inspect", "audit", "tests"],
        "questions": ["Why not randomly split all utterances?", "What does speaker-disjointness prevent?", "Why inspect before parsing?", "How can you prove which trials were used?", "Why are protocol hashes not proof of official provenance?"],
    },
    2: {
        "title": "Features and transparent baseline",
        "built": "Validated mono audio reader, fixed log-spectral mean/std frontend, train-only standardizer, seeded logistic learner, CPU/CUDA auto selection, and persisted training history/model checksum.",
        "why": "Simple features and explicit gradients are defensible starting points. Whole-clip pooling avoids arbitrary cropping; fixed-size features keep the classifier inexpensive. Train-only normalization avoids leakage. Optional CUDA accelerates fitting without requiring GPU hardware.",
        "learn": "Derive binary cross-entropy plus L2 and the sigmoid gradient. Explain FFT resolution, log floors, constant feature handling, temporal pooling losses, imbalance, and why sigmoid scores are not calibrated probabilities.",
        "stages": ["train", "tests"],
        "questions": ["Why use a simple baseline before a deep network?", "Which temporal cues does pooling discard?", "Why is normalization fit only on train?", "How does L2 affect the gradient?", "Which operations actually run on CUDA?"],
    },
    3: {
        "title": "Dev selection and held-out evaluation",
        "built": "Spoof-positive metrics, empirical EER selection on dev, frozen eval threshold, raw score CSVs with audio hashes, per-attack train-membership flags, and CSV-to-JSON metric verification.",
        "why": "Eval operating-point selection leaks evaluation labels into deployment decisions. A fixed clean-dev threshold shows threshold transfer honestly. Per-attack miss rates expose unseen-attack weaknesses without invalid one-class ROC calculations.",
        "learn": "Trace score >= threshold through every confusion-matrix cell. FAR here means bonafide flagged as spoof and FRR means spoof missed, not an unstated ASV acceptance convention. Understand the finite empirical EER approximation and why it differs from official interpolated tooling.",
        "stages": ["evaluate", "score_verification", "tests"],
        "questions": ["Why can eval EER be reported but not deployed as the threshold?", "What changes if spoof becomes label zero?", "Why can one attack not have standalone ROC-AUC?", "What makes an attack unseen in this code?", "Why is accuracy insufficient under imbalance?"],
    },
    4: {
        "title": "Robustness, honest reporting and handoff",
        "built": "Deterministic waveform-SNR noise, real MP3 encode/decode, condition-level failures, JSON-derived reports, offline demo, Colab notebook and checksum-manifest source ZIP packaging.",
        "why": "Noise and compression stress distinct distribution shifts. A fixed clean-dev threshold avoids optimistic condition-specific tuning. The initial missing-FFmpeg failure was preserved; a pinned executable-wheel fallback corrected that environment problem. No bad result is silently replaced.",
        "learn": "Derive signal/noise power scaling in decibels. Explain the effects of codec filtering, nonlinearities and delay. Distinguish synthetic software validation from speech benchmark evidence. Discuss real-noise limitations and paired clean/degraded evaluation at identical trial scope.",
        "stages": ["robustness", "report", "tests"],
        "questions": ["Why retain the clean-dev threshold under noise?", "Why is quantization not an MP3 experiment?", "How are noise draws reproducible independent of ordering?", "Why avoid clipping additive-noise audio?", "What remains unproven after synthetic smoke tests pass?"],
    },
}


def main():
    source = Path("results/smoke/status.json")
    status = json.loads(source.read_text(encoding="utf-8"))
    Path("docs/validation.json").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    by_stage = {item["stage"]: item for item in status["stages"]}
    for number, phase in PHASES.items():
        lines = [f"# Phase {number}: {phase['title']}", "", "## What was built", phase["built"],
                 "", "## Why these decisions", phase["why"], "", "## What to learn and explain", phase["learn"],
                 "", "## Smoke test (run before writing this note)",
                 "Source: `docs/validation.json`, copied from `results/smoke/status.json` by this script.",
                 "Scope: synthetic tone fixtures only, NOT ASVspoof speech or benchmark performance."]
        for stage in phase["stages"]:
            record = by_stage[stage]
            lines.append(f"- `{stage}`: {'PASS' if record['exit_code'] == 0 else 'FAIL'}; see recorded stdout/stderr.")
        lines += ["", "## Honest limitations", "Official LA data was unavailable. Real-file format, official speaker-disjointness, full-data training, unseen-attack effectiveness and real robustness remain unverified. CUDA, Windows and Colab execution remain untested in this sandbox.",
                  "The original missing-codec failure is in `docs/initial_smoke_failure.json`; do not delete failed attempts when conducting experiments.",
                  "", "## Viva questions"]
        lines += [f"{i}. {question}" for i, question in enumerate(phase["questions"], 1)]
        Path(f"docs/phase_{number}_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Phase notes and synthetic validation evidence written.")


if __name__ == "__main__":
    main()
