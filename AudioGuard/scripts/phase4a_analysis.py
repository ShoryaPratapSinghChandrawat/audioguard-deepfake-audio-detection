"""Generate Phase 4A analyses from frozen scores and TRAIN-only nuisance cues."""
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment import load_model
from src.common import load_config
from src.data.protocol import load_dataset
from src.evaluation.metrics import empirical_eer, metrics
from src.features.acoustic import extract
from src.features.spectrum import read_audio
from src.robustness.shortcuts import nuisance

OUT = ROOT / "results/phase4a"
CNN_DIR = ROOT / "results/real_cnn_logmel/cnn_logmel"
SVM_DIR = ROOT / "results/real_svm/phase2_original/svm_mfcc"
KNOWN_REUSED = {"A16": "A04", "A19": "A06"}
UNKNOWN = [*(f"A{i:02d}" for i in range(7, 16)), "A17", "A18"]


def score_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return {r["utterance"]: {**r, "label": int(r["label"]), "score": float(r["score"])}
                for r in csv.DictReader(stream)}


def dump_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def auc_eer(y, scores, threshold):
    y, scores = np.asarray(y, dtype=np.int64), np.asarray(scores, dtype=np.float64)
    detail = metrics(y, scores, threshold)
    return {**detail, "roc_auc": float(roc_auc_score(y, scores)),
            "eer": empirical_eer(y, scores), "spoof_recall": float(detail["tp"]/(detail["tp"]+detail["fn"])),
            "bona_fide_fpr": float(detail["fp"]/(detail["tn"]+detail["fp"]))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cnn_meta = json.loads((CNN_DIR / "training.json").read_text(encoding="utf-8"))
    svm_meta = json.loads((SVM_DIR / "training.json").read_text(encoding="utf-8"))
    cfg = cnn_meta["config"]
    data, audit = load_dataset(cfg)
    eval_rows = data["eval"]
    cnn_csv = score_rows(CNN_DIR / "eval_clean_scores.csv")
    svm_csv = score_rows(SVM_DIR / "eval_clean_scores.csv")
    utterances = {r["utterance"] for r in eval_rows}
    if set(cnn_csv) != utterances or set(svm_csv) != utterances:
        raise ValueError("EVAL score utterances do not equal the official protocol")
    for row in eval_rows:
        if cnn_csv[row["utterance"]]["label"] != row["label"] or svm_csv[row["utterance"]]["label"] != row["label"]:
            raise ValueError("Saved EVAL score label mismatch")

    model_metas = {"CNN": cnn_meta, "SVM": svm_meta}
    score_map = {"CNN": cnn_csv, "SVM": svm_csv}
    thresholds = {name: float(meta["threshold_selection"]["threshold"]) for name, meta in model_metas.items()}
    all_rows = [*data["train"], *data["dev"], *data["eval"]]
    cues, durations = {}, {}
    cue_start = time.perf_counter()
    frontend = cfg["frontend"]
    for index, row in enumerate(all_rows, 1):
        audio = read_audio(row["path"], frontend["sample_rate"])
        feat = nuisance(audio, frontend["sample_rate"], cfg["shortcut"]["silence_dbfs"])
        cues[row["utterance"]] = feat
        durations[row["utterance"]] = float(len(audio) / frontend["sample_rate"])
        if index % 10000 == 0:
            print(f"CUE_EXTRACTION_PROGRESS {index}/{len(all_rows)}", flush=True)
    cue_runtime = time.perf_counter() - cue_start

    shortcut_cfg = cfg["shortcut"]
    shortcut = make_pipeline(StandardScaler(), LogisticRegression(
        C=shortcut_cfg["C"], max_iter=shortcut_cfg["max_iter"],
        class_weight=shortcut_cfg["class_weight"], random_state=cfg["seed"]))
    shortcut.fit(np.stack([cues[r["utterance"]] for r in data["train"]]),
                 np.asarray([r["label"] for r in data["train"]]))
    dev_scores = shortcut.decision_function(np.stack([cues[r["utterance"]] for r in data["dev"]]))
    shortcut_threshold = empirical_eer([r["label"] for r in data["dev"]], dev_scores)["threshold"]
    eval_shortcut_scores = shortcut.decision_function(np.stack([cues[r["utterance"]] for r in eval_rows]))
    shortcut_eval = auc_eer([r["label"] for r in eval_rows], eval_shortcut_scores, shortcut_threshold)
    shortcut_dev = auc_eer([r["label"] for r in data["dev"]], dev_scores, shortcut_threshold)
    shortcut_runtime = {"all_partition_cue_extraction_seconds": cue_runtime,
                        "fit_dev_selection_eval_seconds_including_cue_arrays": time.perf_counter()-cue_start,
                        "threshold_selected_on_dev": float(shortcut_threshold),
                        "features": ["duration_seconds", "leading_silence_seconds_at_-40_dBFS", "waveform_RMS"],
                        "results": {"dev": shortcut_dev, "eval": shortcut_eval}}
    dump_json(OUT / "shortcut_baseline.json", shortcut_runtime)

    model_eval = {}
    per_speaker = {}
    duration_correlations = {}
    group_metrics = {}
    attack_metrics = {}
    for name in ("CNN", "SVM"):
        y = np.asarray([r["label"] for r in eval_rows])
        scores = np.asarray([score_map[name][r["utterance"]]["score"] for r in eval_rows])
        threshold = thresholds[name]
        model_eval[name] = auc_eer(y, scores, threshold)
        duration = np.asarray([durations[r["utterance"]] for r in eval_rows])
        duration_correlations[name] = {}
        for label, label_name in ((0, "bona_fide"), (1, "spoof")):
            mask = y == label
            rho, pvalue = spearmanr(scores[mask], duration[mask])
            duration_correlations[name][label_name] = {
                "n": int(mask.sum()), "spearman_rho": float(rho), "p_value": float(pvalue)}

        by_speaker = defaultdict(list)
        for idx, row in enumerate(eval_rows):
            by_speaker[row["speaker"]].append(idx)
        per_speaker[name] = {}
        for speaker, idxs in sorted(by_speaker.items()):
            speaker_y, speaker_s = y[idxs], scores[idxs]
            bf = speaker_y == 0
            sp = speaker_y == 1
            fp = int(np.sum((speaker_s >= threshold) & bf))
            fn = int(np.sum((speaker_s < threshold) & sp))
            per_speaker[name][speaker] = {
                "bona_fide_n": int(bf.sum()), "bona_fide_fp": fp,
                "bona_fide_fpr": None if not bf.any() else float(fp / bf.sum()),
                "spoof_n": int(sp.sum()), "spoof_fn": fn,
                "spoof_miss_rate": None if not sp.any() else float(fn / sp.sum())}

        attack_metrics[name] = {}
        for attack in sorted({r["attack"] for r in eval_rows if r["label"] == 1}):
            mask = np.asarray([(r["label"] == 0 or r["attack"] == attack) for r in eval_rows])
            spoof_mask = np.asarray([(r["label"] == 1 and r["attack"] == attack) for r in eval_rows])
            sub_y, sub_scores = y[mask], scores[mask]
            attack_metrics[name][attack] = {
                "n_spoof": int(spoof_mask.sum()), "shared_bona_fide_n": int((y == 0).sum()),
                "eer": empirical_eer(sub_y, sub_scores),
                "misses": int(np.sum((scores < threshold) & spoof_mask)),
                "spoof_recall": float(np.mean(scores[spoof_mask] >= threshold)),
                "miss_rate": float(np.mean(scores[spoof_mask] < threshold))}

        for group, attacks in (("unknown_11", UNKNOWN), ("known_reused_A16_A19", list(KNOWN_REUSED))):
            mask = np.asarray([(r["label"] == 0 or r["attack"] in attacks) for r in eval_rows])
            spoof_mask = np.asarray([(r["label"] == 1 and r["attack"] in attacks) for r in eval_rows])
            sub_y, sub_scores = y[mask], scores[mask]
            group_metrics.setdefault(name, {})[group] = {
                "attacks": attacks, "n_bona_fide": int((sub_y == 0).sum()),
                "n_spoof": int(spoof_mask.sum()), "eer": empirical_eer(sub_y, sub_scores),
                "spoof_recall_at_frozen_threshold": float(np.mean(scores[spoof_mask] >= threshold)),
                "miss_rate_at_frozen_threshold": float(np.mean(scores[spoof_mask] < threshold))}

    cnn_threshold = thresholds["CNN"]
    y_eval = np.asarray([r["label"] for r in eval_rows])
    cnn_scores = np.asarray([cnn_csv[r["utterance"]]["score"] for r in eval_rows])
    fp_indices = [i for i, row in enumerate(eval_rows)
                  if row["label"] == 0 and cnn_scores[i] >= cnn_threshold]
    fn_indices = [i for i, row in enumerate(eval_rows)
                  if row["label"] == 1 and cnn_scores[i] < cnn_threshold]
    false_positives = []
    for i in fp_indices:
        row = eval_rows[i]
        false_positives.append({"file_id": row["utterance"], "speaker": row["speaker"],
                                "duration_seconds": durations[row["utterance"]],
                                "score": float(cnn_scores[i])})
    false_negatives = []
    for i in sorted(fn_indices, key=lambda i: cnn_scores[i])[:30]:
        row = eval_rows[i]
        false_negatives.append({"file_id": row["utterance"], "attack": row["attack"],
                                "score": float(cnn_scores[i]),
                                "distance_below_threshold": float(cnn_threshold-cnn_scores[i])})

    # Select three bona fide FPs and three lowest-scoring spoof FNs for illustrative plots.
    representative = []
    for i in sorted(fp_indices, key=lambda i: cnn_scores[i], reverse=True)[:3]:
        representative.append((i, "false_positive"))
    for i in sorted(fn_indices, key=lambda i: cnn_scores[i])[:3]:
        representative.append((i, "false_negative"))
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)
    for ordinal, (i, kind) in enumerate(representative, 1):
        row = eval_rows[i]
        wave = read_audio(row["path"], frontend["sample_rate"])
        logmel = extract(wave, frontend, "logmel")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.imshow(logmel, origin="lower", aspect="auto", interpolation="nearest",
                  extent=[0, len(wave)/frontend["sample_rate"], 0, frontend["sample_rate"]/2],
                  cmap="magma")
        ax.set(title=f"{kind}: {row['utterance']} ({row['attack']}) score={cnn_scores[i]:.5f}",
               xlabel="Time (s)", ylabel="Frequency (Hz)")
        fig.tight_layout()
        fig.savefig(figdir / f"{ordinal:02d}_{kind}_{row['utterance']}.png", dpi=140)
        plt.close(fig)

    result = {
        "status": "completed", "dataset_audit_sha256": hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest(),
        "taxonomy": {"unknown": UNKNOWN, "reused_known": KNOWN_REUSED,
                     "source": "Wang et al., ASVspoof 2019 database paper, Section 3.1, Table 1 and system details"},
        "thresholds": {k: repr(v) for k, v in thresholds.items()},
        "full_eval_metrics": model_eval, "unknown_vs_reused": group_metrics,
        "per_attack": attack_metrics, "per_speaker": per_speaker,
        "duration_spearman": duration_correlations,
        "cnn_bona_fide_false_positives": false_positives,
        "cnn_top30_spoof_false_negatives_lowest_scores": false_negatives,
        "cnn_representative_spectrograms": [p.name for p in sorted(figdir.glob("*.png"))],
        "shortcut_baseline": shortcut_runtime,
        "runtime_seconds": {"all_partition_cue_extraction": cue_runtime,
                             "phase4a_analysis_total": time.perf_counter()-cue_start},
    }
    dump_json(OUT / "phase4a_metrics.json", result)

    def pct(x):
        return "N/A" if x is None else f"{100*x:.3f}%"
    lines = ["# Phase 4A: saved-score analysis", "",
             "All CNN/SVM evaluations use frozen DEV thresholds and saved clean EVAL scores. The shortcut model was fit on TRAIN nuisance cues; its threshold was selected on DEV only.", "",
             "## Attack taxonomy", "",
             "| Classification | Attack IDs | Interpretation |", "|---|---|---|",
             "| Unknown systems | A07–A15, A17, A18 | 11 unknown ASVspoof systems |",
             "| Known reused references | A16, A19 | A16 reuses A04's algorithm; A19 reuses A06's algorithm |", "",
             "The Wang et al. database paper states that A07–A19 excluding A16/A19 are the 11 unknown systems, and identifies A16/A04 and A19/A06 algorithm reuse. It also states A17 uses the same VAE-based framework as A05 but a generalized direct waveform-modification method instead of WORLD. This describes generalization to unseen ASVspoof systems only; it is not universal or out-of-dataset generalization. [Wang et al. (2020), paper PDF](https://www.research.ed.ac.uk/files/157030755/ASVspoof_2019_WANG_DOA08052020_AFV.pdf).", "",
             "## Unknown systems and reused references", "",
             "| Model | Group | Attacks | Bona fide / spoof N | Pooled EER | Spoof recall @ frozen threshold | Miss rate @ frozen threshold |", "|---|---|---|---:|---:|---:|---:|"]
    for model in ("CNN", "SVM"):
        for group in ("unknown_11", "known_reused_A16_A19"):
            g = group_metrics[model][group]
            lines.append(f"| {model} | {group} | {', '.join(g['attacks'])} | {g['n_bona_fide']} / {g['n_spoof']} | {pct(g['eer']['eer'])} | {pct(g['spoof_recall_at_frozen_threshold'])} | {pct(g['miss_rate_at_frozen_threshold'])} |")
    lines += ["", "## Per-speaker EVAL error rates", "", "FPR denominator is that speaker's bona fide trials; miss denominator is that speaker's spoof trials. N/A indicates no trials of that class.", "",
              "| Speaker | BF FP / N | BF FPR CNN | Spoof FN / N | Spoof miss CNN | BF FPR SVM | Spoof miss SVM |", "|---|---:|---:|---:|---:|---:|---:|"]
    for speaker in sorted(per_speaker["CNN"]):
        c, s = per_speaker["CNN"][speaker], per_speaker["SVM"][speaker]
        lines.append(f"| {speaker} | {c['bona_fide_fp']} / {c['bona_fide_n']} | {pct(c['bona_fide_fpr'])} | {c['spoof_fn']} / {c['spoof_n']} | {pct(c['spoof_miss_rate'])} | {pct(s['bona_fide_fpr'])} | {pct(s['spoof_miss_rate'])} |")
    for model in ("CNN", "SVM"):
        fprs = [v['bona_fide_fpr'] for v in per_speaker[model].values() if v['bona_fide_fpr'] is not None]
        misses = [v['spoof_miss_rate'] for v in per_speaker[model].values() if v['spoof_miss_rate'] is not None]
        fp_speakers = sum(v['bona_fide_fp'] > 0 for v in per_speaker[model].values())
        spoof_speakers = sum(v['spoof_n'] > 0 for v in per_speaker[model].values())
        fn_speakers = sum(v['spoof_fn'] > 0 for v in per_speaker[model].values())
        lines.append(f"\n{model}: speaker FPR range {min(fprs):.4%}–{max(fprs):.4%}; speaker miss-rate range {min(misses):.4%}–{max(misses):.4%}. Bona fide false positives occur for {fp_speakers}/{len(per_speaker[model])} speakers; spoof misses occur for {fn_speakers}/{spoof_speakers} speakers with spoof samples. This summarizes spread only; it does not establish identity reliance.")
    lines += ["", "## CNN error review", "", "### All bona fide false positives", "",
              "| file_id | speaker | duration (s) | raw logit |", "|---|---|---:|---:|"]
    for e in false_positives:
        lines.append(f"| {e['file_id']} | {e['speaker']} | {e['duration_seconds']:.4f} | {e['score']:.8f} |")
    lines += ["", "### 30 lowest-scoring spoof false negatives", "",
              "These are the lowest spoof logits among samples below threshold, not the samples nearest threshold.", "",
              "| file_id | attack | score | distance below frozen threshold |", "|---|---|---:|---:|"]
    for e in false_negatives:
        lines.append(f"| {e['file_id']} | {e['attack']} | {e['score']:.8f} | {e['distance_below_threshold']:.8f} |")
    lines += ["", "### Representative CNN Log-Mel plots", ""]
    lines += [f"![{name}](figures/{name})" for name in result["cnn_representative_spectrograms"]]
    lines += ["", "Six plots are descriptive examples only; they do not establish a general acoustic cause.", "",
              "## Duration correlation", "", "Spearman score-duration correlations are descriptive within each class.", "",
              "| Model | Class | N | Spearman rho | p-value |", "|---|---|---:|---:|---:|"]
    for model in ("CNN", "SVM"):
        for cls, val in duration_correlations[model].items():
            lines.append(f"| {model} | {cls} | {val['n']} | {val['spearman_rho']:.6f} | {val['p_value']:.6g} |")
    lines += ["", "## Nuisance-cue shortcut baseline", "",
              "Features are duration, leading silence at −40 dBFS, and waveform RMS, computed with the repository's `nuisance` implementation. Logistic regression was trained on TRAIN only; threshold selection used DEV. A materially above-chance result means these cues contain predictive information in this dataset. It does not show that the CNN or SVM relies on them; model-specific counterfactual/ablation evidence is required.", "",
              "| Partition | AUC | EER | Threshold | FPR @ DEV threshold | FNR @ DEV threshold |", "|---|---:|---:|---:|---:|---:|"]
    for part, obj in (("DEV", shortcut_dev), ("EVAL", shortcut_eval)):
        lines.append(f"| {part} | {obj['roc_auc']:.6f} | {pct(obj['eer']['eer'])} | {shortcut_threshold:.12g} | {pct(obj['far'])} | {pct(obj['frr'])} |")
    lines.append(f"\nCue extraction across all three partitions took {cue_runtime:.3f} seconds; fit, DEV selection, and EVAL scoring are recorded in `shortcut_baseline.json`.")
    (OUT / "phase4a_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "runtime_seconds": result["runtime_seconds"],
                      "fp_count": len(false_positives), "fn_count": len(fn_indices),
                      "group_metrics": group_metrics, "duration_spearman": duration_correlations,
                      "shortcut": shortcut_runtime}, indent=2), flush=True)


if __name__ == "__main__":
    main()
