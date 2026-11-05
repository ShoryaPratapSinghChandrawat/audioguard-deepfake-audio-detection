"""Render README validation status and phase notes only after saved smoke evidence.

Numeric evidence is taken verbatim from the script-generated validation JSON, not
invented in prose. Technical notes explain the decisions and remaining limitations.
"""
import json
from pathlib import Path

PHASES = {
    5: ("Cached log-Mel and MFCC frontends",
        "Implemented configured HTK log-Mel, DCT MFCC, temporal regression deltas, pooling and atomic disk caches.",
        "A shared frontend enables controlled input ablation; content-addressing prevents stale settings or changed audio from reusing old features.",
        "Explain STFT normalization, log-plus-epsilon, mel spacing, DCT and derivative boundaries. Contrast frame matrices with pooled vectors.",
        ["Why include waveform bytes in a cache key?", "How is log-plus-epsilon different from log flooring?", "Why retain delta and delta-delta?", "What does pooling lose?", "Why record mel-filter normalization?" ]),
    6: ("CNN fitting and train-only sanity gate",
        "Implemented configurable convolution blocks, weighted BCE, AdamW, dev-loss early stopping, restored checkpoints and train/dev curves.",
        "The train-only memorization gate finds plumbing failures before normal fitting; its weights are discarded. Train-only normalization and dev-only selection avoid leakage.",
        "Derive weighted BCE, distinguish regularization in sanity versus real fitting, and explain global pooling, BatchNorm buffers and center-crop limitations.",
        ["Why must the sanity subset exclude dev?", "How is pos_weight calculated?", "Why restart after memorization?", "Which checkpoint is retained?", "What confounds a CNN-versus-SVM comparison?" ]),
    7: ("Ablation evaluation, shortcuts and error audits",
        "Implemented fixed-threshold ablation evaluation, diagnostic per-attack EER, per-speaker errors, ROC/confusion plots, ranked errors and silence/nuisance controls.",
        "Eval thresholds cannot be retuned. Shared bonafide pools make attack-specific ROC meaningful. Nuisance baselines probe possible shortcuts without proving causation.",
        "Trace precision/recall orientation, explain null class-specific speaker rates and distinguish unseen attack IDs from unseen synthesis families. Never infer method reuse from IDs.",
        ["Which samples enter per-attack EER?", "Why is a speaker's FAR sometimes null?", "Can protocols prove synthesis reuse?", "Why retain the clean-dev threshold after trimming?", "How should high-confidence errors inform a held-out study?" ]),
    8: ("Results-backed demo and source ZIP handoff",
        "Implemented a Gradio upload demo, JSON-backed evidence panel, limitations notice, glossary, research guide and integrity-checked complete source ZIP.",
        "The demo must expose scope and unknown performance instead of hardcoded scores. The ZIP packages real files while excluding datasets, environments and weights.",
        "Distinguish score from calibrated confidence, inspect saved provenance and understand why a UI patch-export button cannot be changed by repository files.",
        ["Where do displayed metrics originate?", "What if eval results are missing?", "Why not load an unknown joblib artifact?", "Which files belong outside the source ZIP?", "What remains unknown after synthetic tests pass?" ]),
}


def main():
    path = Path("docs/research_validation.json")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    status_lines = ["Scope: " + evidence["scope"], ""]
    for record in evidence["records"]:
        name = " ".join(record["command"][1:])
        status_lines.append("- `" + name + "`: " + ("PASS" if record["exit_code"] == 0 else "FAIL") + ".")
        if "pytest" in record["command"]:
            status_lines += ["", "```text", record["stdout_tail"].strip(), "```", ""]
    status_lines.append("Evidence source: `docs/research_validation.json`. No official-data metrics are available.")
    readme = Path("README.md")
    text = readme.read_text(encoding="utf-8")
    before, remainder = text.split("<!-- VALIDATION:START -->", 1)
    _, after = remainder.split("<!-- VALIDATION:END -->", 1)
    readme.write_text(before + "<!-- VALIDATION:START -->\n" + "\n".join(status_lines) + "\n<!-- VALIDATION:END -->" + after, encoding="utf-8")
    for number, (title, built, why, learn, questions) in PHASES.items():
        lines = [f"# Phase {number}: {title}", "", "## What was built", built, "", "## Why", why,
                 "", "## What to learn and explain", learn, "", "## Smoke test before this note", *status_lines,
                 "", "## Honest limitations", "Only synthetic tone fixtures were executed here. Official-data effectiveness, taxonomy and full-data resource usage remain unverified. CUDA, Windows and Colab were not exercised.",
                 "", "## Viva questions", *[f"{i}. {q}" for i, q in enumerate(questions, 1)]]
        Path(f"docs/phase_{number}_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for number in range(1, 5):
        note = Path(f"docs/phase_{number}_notes.md")
        text = note.read_text(encoding="utf-8").split("\n## Research extension update")[0]
        text += "\n## Research extension update\nThe research suite now extends this baseline; see `research_guide.md`, later phase notes and `research_validation.json`. Existing protocol/leakage checks and clean-dev threshold policy are retained. Synthetic validation does not establish official-data performance.\n"
        note.write_text(text, encoding="utf-8")
    print("Updated README and phase notes from saved research validation evidence.")


if __name__ == "__main__":
    main()
