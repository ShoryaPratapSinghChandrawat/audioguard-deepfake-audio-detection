"""Fixed-subset robustness inference for saved CNN and SVM checkpoints."""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment import load_model
from src.common import load_config
from src.data.protocol import load_dataset
from src.evaluation.metrics import empirical_eer, metrics
from src.features.acoustic import FRONTEND_VERSION, cached, extract, pooled
from src.features.spectrum import read_audio
from src.models.cnn import tensor_input
from src.robustness.shortcuts import trim
from src.robustness.transforms import ffmpeg_executable, mp3, noise

OUT = ROOT / "results/phase4b"
RUN_OUT = OUT / "optimized_results"
MANIFEST = OUT / "subset_manifest.jsonl"
CNN_DIR = ROOT / "results/real_cnn_logmel/cnn_logmel"
SVM_REFERENCE = ROOT / "results/real_svm/phase2_original/svm_mfcc"
SVM_ARTIFACTS = ROOT / "results/phase4a/artifact_inputs/svm_mfcc"
CACHE_ROOT = ROOT / "data/feature_cache_phase4b_optimized_v3"
WAVE_ROOT = OUT / "optimized_preflight_v3/waveforms"
N_DEBUG = 500
N_SUBSET = 10000
MAX_RUNTIME_SECONDS = 3 * 60 * 60
RUNTIME_SAFETY_FACTOR = 1.25


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_scores(path):
    with path.open(newline="", encoding="utf-8") as f:
        return {r["utterance"]: float(r["score"]) for r in csv.DictReader(f)}


def metric_report(rows, scores, threshold):
    y = np.asarray([r["label"] for r in rows], dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    base = metrics(y, s, threshold)
    per_attack = {}
    for attack in sorted({r["attack"] for r in rows if r["label"] == 1}):
        mask = np.asarray([r["label"] == 0 or r["attack"] == attack for r in rows])
        spoof = np.asarray([r["label"] == 1 and r["attack"] == attack for r in rows])
        per_attack[attack] = {
            "n_spoof": int(spoof.sum()), "shared_bona_fide_n": int(np.sum(y == 0)),
            "eer": empirical_eer(y[mask], s[mask]),
            "miss_rate": float(np.mean(s[spoof] < threshold)),
            "spoof_recall": float(np.mean(s[spoof] >= threshold)),
        }
    return {**base, "balanced_accuracy": float(balanced_accuracy_score(y, s >= threshold)),
            "spoof_recall": base["tp"] / (base["tp"] + base["fn"]),
            "bona_fide_fpr": base["fp"] / (base["fp"] + base["tn"]),
            "pooled_eer": empirical_eer(y, s), "per_attack": per_attack}


def condition_transform(name, audio, row, sr, seed, silence_dbfs):
    if name.startswith("noise_"):
        snr = int(name.split("_")[1].removesuffix("dB"))
        return noise(audio, snr, seed, row["utterance"])
    if name.startswith("mp3_"):
        bitrate = int(name.split("_")[1].removesuffix("kbps"))
        return mp3(audio, sr, bitrate)
    if name == "trimmed":
        return trim(audio, silence_dbfs)
    raise ValueError(f"Unknown condition {name}")


def wave_paths(condition):
    folder = WAVE_ROOT / condition
    folder.mkdir(parents=True, exist_ok=True)
    return folder, folder / "waveforms.jsonl"


def ensure_wave(row, condition, cfg, existing):
    folder, manifest_path = wave_paths(condition)
    target = folder / f"{row['utterance']}.wav"
    known_sha = existing.get(row["utterance"])
    if target.exists():
        if not known_sha:
            raise RuntimeError(f"Unverified partial waveform exists; preserve and stop: {target}")
        actual_sha = sha256(target)
        if actual_sha != known_sha:
            raise RuntimeError(f"Saved waveform SHA-256 mismatch: {target}")
        sr = int(cfg["frontend"]["sample_rate"])
        waveform = read_audio(target, sr)
        return target, waveform, {"source_decode_seconds": 0.0,
                                  "transform_encode_decode_seconds": 0.0,
                                  "waveform_disk_seconds": 0.0,
                                  "waveform_validation_seconds": 0.0}, None, actual_sha
    if known_sha:
        raise RuntimeError(f"Waveform manifest refers to missing file: {target}")
    sr = int(cfg["frontend"]["sample_rate"])
    timing = {"source_decode_seconds": 0.0,
              "transform_encode_decode_seconds": 0.0,
              "waveform_disk_seconds": 0.0,
              "waveform_validation_seconds": 0.0}
    t0 = time.perf_counter()
    original = read_audio(row["path"], sr)
    timing["source_decode_seconds"] = time.perf_counter()-t0
    t0 = time.perf_counter()
    transformed = condition_transform(condition, original, row, sr, cfg["analysis"]["seed"],
                                      cfg["analysis"]["trim_dbfs"])
    timing["transform_encode_decode_seconds"] = time.perf_counter()-t0
    measured_snr = None
    if condition.startswith("noise_"):
        asked = int(condition.split("_")[1].removesuffix("dB"))
        measured_snr = float(10*np.log10(np.mean(original.astype(np.float64)**2) /
                                          np.mean((transformed-original).astype(np.float64)**2)))
        if abs(measured_snr-asked) > 0.001:
            raise ValueError(f"Noise SNR verification failed for {row['utterance']}: {measured_snr}")
    if transformed.ndim != 1 or not len(transformed) or not np.isfinite(transformed).all():
        raise ValueError(f"Invalid transformed waveform for {row['utterance']}")
    temporary = target.with_suffix(".wav.tmp")
    t0 = time.perf_counter()
    sf.write(temporary, transformed, sr, subtype="FLOAT", format="WAV")
    os.replace(temporary, target)
    timing["waveform_disk_seconds"] = time.perf_counter()-t0
    t0 = time.perf_counter()
    actual_sha = sha256(target)
    value = {"utterance": row["utterance"], "sha256": actual_sha,
             "sample_rate": sr, "samples": len(transformed)}
    timing["waveform_validation_seconds"] = time.perf_counter()-t0
    with manifest_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, sort_keys=True)+"\n")
        f.flush()
        os.fsync(f.fileno())
    existing[row["utterance"]] = value["sha256"]
    return target, transformed, timing, measured_snr, value["sha256"]


def cached_from_memory(path, waveform, audio_sha256, cfg, kind):
    """Match acoustic.cached identity while avoiding a redundant WAV decode."""
    identity={"audio_sha256":audio_sha256,
              "config":{k:v for k,v in cfg.items() if k!="cache_dir"},
              "kind":kind,"implementation":FRONTEND_VERSION}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    folder=Path(cfg["cache_dir"]); folder.mkdir(parents=True,exist_ok=True)
    target=folder/(key+".npy")
    expected=cfg["n_mels"] if kind=="logmel" else cfg["n_mfcc"]*3
    if target.exists():
        value=np.load(target,allow_pickle=False)
        if value.ndim!=2 or value.shape[0]!=expected or not np.isfinite(value).all():
            raise ValueError(f"Invalid isolated feature cache entry: {target}")
        return value,True
    value=extract(waveform,cfg,kind)
    with tempfile.NamedTemporaryFile(dir=folder,suffix=".npy",delete=False) as f:
        tmp=Path(f.name); np.save(f,value,allow_pickle=False)
    try: os.replace(tmp,target)
    finally: tmp.unlink(missing_ok=True)
    return value,False


def read_wave_manifest(condition):
    _, path = wave_paths(condition)
    if not path.exists():
        return {}
    result = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["utterance"] in result:
                raise ValueError(f"Duplicate waveform manifest entry: {item['utterance']}")
            result[item["utterance"]] = item["sha256"]
    return result


def process_condition(rows, condition, cfg, cnn_model, cnn_scaler, cnn_device, svm_model,
                      svm_cfg, debug=False):
    total_start = time.perf_counter()
    existing = read_wave_manifest(condition)
    wave_seconds, cnn_feature_seconds, svm_feature_seconds = 0., 0., 0.
    wave_parts = {k: 0.0 for k in ("source_decode_seconds", "transform_encode_decode_seconds",
                                   "waveform_disk_seconds", "waveform_validation_seconds")}
    cnn_infer_seconds, svm_infer_seconds = 0., 0.
    measured_snr_values = []
    cnn_values, svm_values = [], []
    validation={"clips_checked":0,"max_waveform_roundtrip_abs_diff":0.0,
                "max_cnn_feature_abs_diff":0.0,"max_svm_feature_abs_diff":0.0,
                "previous_mp3_64_waveform_abs_diff":None,
                "previous_mp3_64_cnn_feature_abs_diff":None,
                "previous_mp3_64_svm_feature_abs_diff":None}
    cnn_frontend = dict(cfg["frontend"], cache_dir=str(CACHE_ROOT / "cnn_logmel" / condition))
    svm_frontend = dict(svm_cfg["frontend"], cache_dir=str(CACHE_ROOT / "svm_mfcc" / condition))
    print(f"CACHE_ISOLATION CNN={cnn_frontend['cache_dir']}", flush=True)
    print(f"CACHE_ISOLATION SVM={svm_frontend['cache_dir']}", flush=True)
    for start in range(0, len(rows), 32):
        chunk = rows[start:start+32]
        cnn_batch, svm_batch = [], []
        for row in chunk:
            t0 = time.perf_counter()
            wave_path, waveform, wave_timing, measured, wave_sha = ensure_wave(row, condition, cfg, existing)
            for key, value in wave_timing.items():
                wave_parts[key] += value
            wave_seconds += time.perf_counter() - t0
            if measured is not None:
                measured_snr_values.append(measured)
            t0 = time.perf_counter()
            cnn_feature, cnn_hit = cached_from_memory(wave_path,waveform,wave_sha,cnn_frontend,"logmel")
            cnn_batch.append(tensor_input(cnn_feature, cnn_scaler, cfg["cnn"]["frames"]))
            cnn_feature_seconds += time.perf_counter() - t0
            t0 = time.perf_counter()
            svm_matrix, svm_hit = cached_from_memory(wave_path,waveform,wave_sha,svm_frontend,"mfcc")
            svm_feature = pooled(svm_matrix)
            svm_batch.append(svm_feature)
            svm_feature_seconds += time.perf_counter() - t0
            if validation["clips_checked"]<20:
                roundtrip=read_audio(wave_path,cfg["frontend"]["sample_rate"])
                waveform_diff=float(np.max(np.abs(waveform-roundtrip)))
                cnn_reference=extract(roundtrip,cnn_frontend,"logmel")
                svm_reference=extract(roundtrip,svm_frontend,"mfcc")
                validation["max_waveform_roundtrip_abs_diff"]=max(validation["max_waveform_roundtrip_abs_diff"],waveform_diff)
                validation["max_cnn_feature_abs_diff"]=max(validation["max_cnn_feature_abs_diff"],float(np.max(np.abs(cnn_reference-cnn_feature))))
                validation["max_svm_feature_abs_diff"]=max(validation["max_svm_feature_abs_diff"],float(np.max(np.abs(svm_reference-svm_matrix))))
                validation["clips_checked"]+=1
                if condition=="mp3_64kbps":
                    old_wave=WAVE_ROOT.parent.parent/"waveforms"/condition/f"{row['utterance']}.wav"
                    if old_wave.exists():
                        old_audio=read_audio(old_wave,cfg["frontend"]["sample_rate"])
                        validation["previous_mp3_64_waveform_abs_diff"]=max(
                            validation["previous_mp3_64_waveform_abs_diff"] or 0.0,
                            float(np.max(np.abs(waveform-old_audio))))
                        old_cnn_cfg=dict(cfg["frontend"],cache_dir=str(ROOT/"data/feature_cache_phase4b/cnn_logmel/mp3_64kbps"))
                        old_svm_cfg=dict(svm_cfg["frontend"],cache_dir=str(ROOT/"data/feature_cache_phase4b/svm_mfcc/mp3_64kbps"))
                        old_cnn=cached(old_wave,old_cnn_cfg,"logmel")
                        old_svm=cached(old_wave,old_svm_cfg,"mfcc")
                        validation["previous_mp3_64_cnn_feature_abs_diff"]=max(
                            validation["previous_mp3_64_cnn_feature_abs_diff"] or 0.0,
                            float(np.max(np.abs(old_cnn-cnn_feature))))
                        validation["previous_mp3_64_svm_feature_abs_diff"]=max(
                            validation["previous_mp3_64_svm_feature_abs_diff"] or 0.0,
                            float(np.max(np.abs(old_svm-svm_matrix))))
        t0 = time.perf_counter()
        tensor = torch.stack(cnn_batch).to(cnn_device)
        with torch.no_grad():
            cnn_values.extend(cnn_model(tensor).cpu().tolist())
        cnn_infer_seconds += time.perf_counter() - t0
        t0 = time.perf_counter()
        svm_values.extend(svm_model.decision_function(np.stack(svm_batch)).tolist())
        svm_infer_seconds += time.perf_counter() - t0
        if (start+len(chunk)) % 1000 == 0 or start+len(chunk) == len(rows):
            print(f"{condition} PROGRESS {start+len(chunk)}/{len(rows)}", flush=True)
    times = {"wall_seconds": time.perf_counter()-total_start,
             "waveform_generation_or_validation_seconds": wave_seconds,
             "waveform_components": wave_parts,
             "cnn_feature_cache_seconds": cnn_feature_seconds,
             "svm_feature_cache_seconds": svm_feature_seconds,
             "cnn_inference_seconds": cnn_infer_seconds,
             "svm_inference_seconds": svm_infer_seconds}
    if measured_snr_values:
        times["measured_snr_db_min"] = float(min(measured_snr_values))
        times["measured_snr_db_max"] = float(max(measured_snr_values))
        times["measured_snr_samples"] = len(measured_snr_values)
    times["optimization_equivalence"] = validation
    return cnn_values, svm_values, times


def read_clean_reference(rows):
    cnn_ref = load_scores(CNN_DIR / "eval_clean_scores.csv")
    svm_ref = load_scores(SVM_REFERENCE / "eval_clean_scores.csv")
    return {"CNN": [cnn_ref[r["utterance"]] for r in rows],
            "SVM": [svm_ref[r["utterance"]] for r in rows]}


def estimate_storage(rows, cfg, svm_cfg, free_bytes):
    samples = 0
    cnn_bytes_one = 0
    svm_bytes_one = 0
    sr, win, hop = cfg["frontend"]["sample_rate"], cfg["frontend"]["win_length"], cfg["frontend"]["hop_length"]
    for row in rows:
        info = sf.info(row["path"])
        samples += info.frames
        frames = 1 if info.frames < win else 1 + (info.frames-win)//hop
        cnn_bytes_one += cfg["frontend"]["n_mels"]*frames*4+128
        svm_bytes_one += svm_cfg["frontend"]["n_mfcc"]*3*frames*4+128
    waves_all = 6*(samples*4+len(rows)*44)
    features_all = 6*(cnn_bytes_one+svm_bytes_one)
    estimate = waves_all+features_all
    return {"free_disk_bytes_before": int(free_bytes), "subset_wave_samples_clean": samples,
            "estimated_waveform_bytes_all_conditions": waves_all,
            "estimated_feature_cache_bytes_all_conditions": features_all,
            "estimated_total_bytes": estimate,
            "estimated_free_after_bytes": int(free_bytes-estimate),
            "estimate_note": "Uses clean durations and uncompressed float32 WAVs; assumes every condition has clean-length cache arrays, including trimmed. It is conservative for trimmed clips."}


def run_debug(rows, cfg, svm_cfg, models, free_bytes):
    subset_rows = rows[:N_DEBUG]
    result = {"status": "running", "scope": "500-clip timing only; no benchmark metrics",
              "conditions": {}, "clips_per_condition": len(subset_rows),
              "free_disk_preflight": estimate_storage(rows, cfg, svm_cfg, free_bytes),
              "ffmpeg": ffmpeg_executable(), "seed": cfg["analysis"]["seed"]}
    ffmpeg_version = subprocess.run([result["ffmpeg"], "-version"], capture_output=True,
                                    text=True, check=True).stdout.splitlines()[0]
    result["ffmpeg_version"] = ffmpeg_version
    conditions=["noise_20dB","noise_10dB","noise_0dB","mp3_128kbps","mp3_64kbps","trimmed"]
    for condition in conditions:
        _, _, times = process_condition(subset_rows, condition, cfg, *models, svm_cfg, debug=True)
        result["conditions"][condition] = times
    measured=sum(v["wall_seconds"] for v in result["conditions"].values())
    result["measured_seconds_all_six_conditions"] = measured
    result["total_six_conditions_both_models_estimate_seconds"] = measured*(N_SUBSET/N_DEBUG)*RUNTIME_SAFETY_FACTOR+60.0
    result["total_estimate_method"] = "Sum of measured wall times for each of six conditions on the identical first 500 fixed-manifest clips, scaled by 10,000/500 and a 25% safety margin plus 60 seconds for startup/reporting. Every timing includes source decode, the condition transform (including actual FFmpeg MP3 encode and decode), float-WAV persistence and hash/manifest I/O, separate CNN and SVM feature cache lookup/extraction/persistence, and batched inference. The first 20 clips per condition also include equivalence validation, making the timing conservative."
    result["runtime_gate_seconds"] = MAX_RUNTIME_SECONDS
    result["status"] = "pass" if result["total_six_conditions_both_models_estimate_seconds"] <= MAX_RUNTIME_SECONDS else "blocked_runtime_over_3h"
    result["free_disk_bytes_after_debug"] = int(shutil.disk_usage(ROOT).free)
    (OUT / "optimized_debug_timing.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    if result["status"] != "pass":
        storage = result["free_disk_preflight"]
        lines = ["# Phase 4B preflight: BLOCKED by runtime gate", "",
                 "This is a DEBUG ONLY timing result on the already-frozen 10,000-clip subset. No full-subset robustness metrics were computed and no SNR/bitrate condition results are reported.", "",
                 f"- Manifest SHA-256: `{sha256(MANIFEST)}`; subset N={N_SUBSET}, 1,030 bona fide, 8,970 spoof (690 for each A07–A19).",
                 f"- Debug timing: {measured:.3f}s total for six conditions × 500 fixed clips, including waveform generation/MP3 encode-decode, waveform and cache disk I/O, features, and both model inferences.",
                 f"- Conservative all-six-condition/two-model estimate: {result['total_six_conditions_both_models_estimate_seconds']:.3f}s ({result['total_six_conditions_both_models_estimate_seconds']/3600:.3f}h), above the 10,800s gate.",
                 f"- Disk preflight: {storage['free_disk_bytes_before']:,} bytes free; estimated combined waveform/cache requirement {storage['estimated_total_bytes']:,} bytes; estimated free after {storage['estimated_free_after_bytes']:,} bytes.",
                 f"- FFmpeg: `{result['ffmpeg_version']}`.",
                 "- The complete six-condition, 10,000-clip experiment was stopped as required. The fixed manifest was not reduced or resampled.", "",
                 "The user-approved subset size and conditions were retained; full execution stopped because the runtime gate failed.", ""]
        (OUT/"phase4b_summary.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def plot_curves(result):
    figdir = RUN_OUT / "figures"
    figdir.mkdir(exist_ok=True)
    colors = {"CNN": "tab:blue", "SVM": "tab:orange"}
    metrics_to_plot = [("pooled_eer", "Pooled EER"), ("roc_auc", "ROC-AUC"),
                       ("balanced_accuracy", "Balanced accuracy"), ("spoof_recall", "Spoof recall")]
    noise_names = ["noise_0dB", "noise_10dB", "noise_20dB"]
    for family, names, xvals, xlabel, filename in (
        ("noise", noise_names, [0,10,20], "SNR (dB)", "noise_degradation.png"),
        ("mp3", ["mp3_64kbps", "mp3_128kbps"], [64,128], "MP3 bitrate (kbps)", "mp3_degradation.png"),
    ):
        fig, axs = plt.subplots(2, 2, figsize=(10,7))
        for ax, (key, title) in zip(axs.ravel(), metrics_to_plot):
            for model in ("CNN", "SVM"):
                vals = []
                for name in names:
                    value = result["conditions"][name][model][key]
                    vals.append(value["eer"] if key == "pooled_eer" else value)
                ax.plot(xvals, vals, marker="o", color=colors[model], label=model)
            ax.set(title=title, xlabel=xlabel, ylabel=title)
            if family == "noise": ax.invert_xaxis()
            ax.grid(alpha=.25); ax.legend()
        fig.tight_layout(); fig.savefig(figdir/filename,dpi=150); plt.close(fig)


def write_summary(result, destination=None):
    lines=["# Phase 4B: fixed-subset robustness (SUBSET RESULTS)", "",
           f"Subset manifest: `subset_manifest.jsonl` ({result['subset']['n']:,} clips; SHA-256 `{result['subset']['sha256']}`). Fixed before any condition scores were examined; reused unchanged for both models and all conditions.", "",
           "Conditions and models use saved checkpoints with frozen DEV thresholds and TRAIN-fit normalization. No model retraining, threshold retuning, or EVAL-based selection occurred. Perturbed waveforms are generated once per clip/condition and shared by both models. Separate per-model/per-condition feature caches live under `data/feature_cache_phase4b_optimized_v3/`.", "",
           "## Clean reference on the same subset", "",
           "| Model | EER | AUC | Balanced accuracy | Spoof recall | Bona fide FPR |", "|---|---:|---:|---:|---:|---:|"]
    for model in ("CNN","SVM"):
        m=result["clean_reference"][model]
        lines.append(f"| {model} | {m['pooled_eer']['eer']:.4%} | {m['roc_auc']:.6f} | {m['balanced_accuracy']:.6f} | {m['spoof_recall']:.6f} | {m['bona_fide_fpr']:.6f} |")
    lines += ["", "## Conditions", "", "All metrics are subset-only; thresholds were selected on the full DEV split and remain frozen. EER is the nearest finite empirical ROC point; per-attack EER uses the subset's shared bona fide pool.", "",
              "| Condition | Model | Pooled EER | AUC | Balanced accuracy | Spoof recall | Bona fide FPR |", "|---|---|---:|---:|---:|---:|---:|"]
    order=["noise_20dB","noise_10dB","noise_0dB","mp3_128kbps","mp3_64kbps","trimmed"]
    for cond in order:
        for model in ("CNN","SVM"):
            m=result["conditions"][cond][model]
            lines.append(f"| {cond} | {model} | {m['pooled_eer']['eer']:.4%} | {m['roc_auc']:.6f} | {m['balanced_accuracy']:.6f} | {m['spoof_recall']:.6f} | {m['bona_fide_fpr']:.6f} |")
    lines += ["", "## Per-attack EER", "", "| Condition | Model | Attack | Spoof N | Shared bona fide N | EER | Miss rate @ frozen threshold |", "|---|---|---|---:|---:|---:|---:|"]
    for cond in order:
        for model in ("CNN","SVM"):
            for attack, m in result["conditions"][cond][model]["per_attack"].items():
                lines.append(f"| {cond} | {model} | {attack} | {m['n_spoof']} | {m['shared_bona_fide_n']} | {m['eer']['eer']:.4%} | {m['miss_rate']:.4%} |")
    validation=json.loads((RUN_OUT/"phase4b_optimization_validation.json").read_text(encoding="utf-8"))
    lines += ["", "## Optimization validation", "",
              "Transformation round trips were compared with the saved float32 waveforms and both feature frontends on 20 deterministic clips per condition; exact equality was observed (maximum absolute difference 0.0). Batch-size-32 inference was compared directly with the saved Phase 4B scores on the same first 32 fixed-subset trials per condition, matching the original prediction batch size; tolerance is 1e-6 in raw score units.", "",
              "| Condition | CNN vs saved max abs | SVM vs saved max abs |", "|---|---:|---:|---:|"]
    for condition,values in validation["inference_equivalence"].items():
        lines.append(f"| {condition} | {values['cnn_batch_vs_saved_max_abs']:.3g} | {values['svm_batch_vs_saved_max_abs']:.3g} |")
    lines += ["", f"FFmpeg `{validation['ffmpeg_version']}` encoded each MP3 condition with `libmp3lame`, then decoded it back to mono at {validation['sample_rate']} Hz. Over 20 clips per bitrate, the decoded-minus-source sample-count ranges were {validation['mp3_roundtrip']['mp3_64kbps']['sample_count_delta_min']} to {validation['mp3_roundtrip']['mp3_64kbps']['sample_count_delta_max']} at 64 kbps and {validation['mp3_roundtrip']['mp3_128kbps']['sample_count_delta_min']} to {validation['mp3_roundtrip']['mp3_128kbps']['sample_count_delta_max']} at 128 kbps. Mean common-prefix codec error RMS was {validation['mp3_roundtrip']['mp3_64kbps']['mean_common_prefix_error_rms']:.6g} and {validation['mp3_roundtrip']['mp3_128kbps']['mean_common_prefix_error_rms']:.6g}, respectively. These measurements document the actual lossy codec round trip; sample padding/encoder delay was not cropped from the experiment waveforms.", "",
              "Noise SNR was measured per generated clip: the 500-clip preflight plus full-run new generations covered all 10,000 clips per noise condition. Exact minima and maxima are recorded in the condition runtime entries in `phase4b_metrics.json` and `optimized_debug_timing.json`.", "",
              "## Degradation plots", "", "![Noise SNR curves](figures/noise_degradation.png)", "", "![MP3 bitrate curves](figures/mp3_degradation.png)", "",
              "Interpretation is limited to observed values. Additive synthetic noise is not real-world noise; MP3 results concern the tested FFmpeg codec and bitrates. This is one fixed subset and one seed.", "",
              "## Runtime and disk", "", json.dumps(result["debug_preflight"], indent=2), ""]
    destination=destination or RUN_OUT/"phase4b_summary.md"
    destination.write_text("\n".join(lines),encoding="utf-8")


def finalize_saved_results(manifest, cnn_meta, svm_meta):
    """Validate persisted condition CSVs and build aggregate report without inference."""
    conditions=["noise_20dB","noise_10dB","noise_0dB","mp3_128kbps","mp3_64kbps","trimmed"]
    expected={r["utterance"]:r for r in manifest}
    if len(manifest)!=N_SUBSET or len(expected)!=N_SUBSET:
        raise ValueError("Saved-result finalization requires the unchanged 10,000-row subset")
    if sha256(MANIFEST)!=json.loads((OUT/"subset_manifest_metadata.json").read_text(encoding="utf-8"))["manifest_sha256"]:
        raise ValueError("Subset manifest hash does not match its approved metadata")
    if len([r for r in manifest if r["label"]==0])!=1030 or len([r for r in manifest if r["label"]==1])!=8970:
        raise ValueError("Saved-result finalization found unexpected subset labels")
    thresholds={"CNN":float(cnn_meta["threshold_selection"]["threshold"]),
                "SVM":float(svm_meta["threshold_selection"]["threshold"])}
    result={"status":"completed","scope":"SUBSET RESULTS only",
            "manifest_sha256":sha256(MANIFEST),
            "subset":{"n":len(manifest),"bona_fide":1030,"spoof":8970,"seed":2026,"sha256":sha256(MANIFEST)},
            "thresholds_full_precision":{k:repr(v) for k,v in thresholds.items()},
            "debug_preflight":json.loads((OUT/"optimized_debug_timing.json").read_text(encoding="utf-8")),
            "conditions":{},"clean_reference":{}}
    clean=read_clean_reference(manifest)
    for model in ("CNN","SVM"):
        result["clean_reference"][model]=metric_report(manifest,clean[model],thresholds[model])
    saved_scores={}
    for condition in conditions:
        metric_path=RUN_OUT/f"{condition}_metrics.json"
        if not metric_path.exists():
            raise FileNotFoundError(f"Missing completed condition metrics: {metric_path}")
        metrics_for_condition=json.loads(metric_path.read_text(encoding="utf-8"))
        for model in ("CNN","SVM"):
            path=RUN_OUT/f"{condition}_{model.lower()}_scores.csv"
            with path.open(newline="",encoding="utf-8") as f:
                reader=csv.DictReader(f)
                if reader.fieldnames != ["utterance","speaker","label","attack","score"]:
                    raise ValueError(f"Unexpected score CSV columns: {path}")
                records=list(reader)
            ids=[r["utterance"] for r in records]
            if len(ids)!=N_SUBSET or len(set(ids))!=N_SUBSET or ids!=[r["utterance"] for r in manifest]:
                raise ValueError(f"Score CSV trial identity/order mismatch: {path}")
            for record in records:
                original=expected[record["utterance"]]
                if record["speaker"]!=original["speaker"] or int(record["label"])!=original["label"] or record["attack"]!=original["attack"]:
                    raise ValueError(f"Score CSV label/protocol mismatch: {path}, {record['utterance']}")
                if not np.isfinite(float(record["score"])):
                    raise ValueError(f"Nonfinite saved score: {path}, {record['utterance']}")
            saved_scores[(condition,model)]=[float(r["score"]) for r in records[:32]]
            if metrics_for_condition[model]["n"]!=N_SUBSET:
                raise ValueError(f"Condition metric count mismatch: {metric_path} {model}")
        result["conditions"][condition]=metrics_for_condition
    cnn_model,cnn_scaler,cnn_device=load_model(CNN_DIR,cnn_meta)
    svm_model,_,_=load_model(SVM_ARTIFACTS,svm_meta)
    cnn_cfg=cnn_meta["config"]
    svm_cfg=svm_meta["config"]
    inference_equivalence={}
    mp3_roundtrip={}
    eval_root=Path(cnn_cfg["dataset"]["audio_roots"]["eval"])
    sr=int(cnn_cfg["frontend"]["sample_rate"])
    ffmpeg=ffmpeg_executable()
    ffmpeg_version=subprocess.run([ffmpeg,"-version"],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    for condition in conditions:
        cnn_features=[]; svm_features=[]
        for item in manifest[:32]:
            wav=WAVE_ROOT/condition/f"{item['utterance']}.wav"
            audio=read_audio(wav,sr)
            audio_sha=sha256(wav)
            cf,_=cached_from_memory(wav,audio,audio_sha,dict(cnn_cfg["frontend"],cache_dir=str(CACHE_ROOT/"cnn_logmel"/condition)),"logmel")
            sfv,_=cached_from_memory(wav,audio,audio_sha,dict(svm_cfg["frontend"],cache_dir=str(CACHE_ROOT/"svm_mfcc"/condition)),"mfcc")
            cnn_features.append(tensor_input(cf,cnn_scaler,cnn_cfg["cnn"]["frames"]))
            svm_features.append(pooled(sfv))
        batch=torch.stack(cnn_features).to(cnn_device)
        with torch.no_grad():
            cnn_batch_scores=cnn_model(batch).cpu().numpy().astype(np.float64)
        svm_matrix=np.stack(svm_features)
        svm_batch_scores=svm_model.decision_function(svm_matrix)
        cnn_saved=np.asarray(saved_scores[(condition,"CNN")],dtype=np.float64)
        svm_saved=np.asarray(saved_scores[(condition,"SVM")],dtype=np.float64)
        equivalence={"cnn_batch_vs_saved_max_abs":float(np.max(np.abs(cnn_batch_scores-cnn_saved))),
                     "svm_batch_vs_saved_max_abs":float(np.max(np.abs(svm_batch_scores-svm_saved)))}
        if max(equivalence.values())>1e-6:
            raise ValueError(f"Batch/saved inference equivalence exceeded 1e-6 for {condition}: {equivalence}")
        inference_equivalence[condition]=equivalence
        if condition.startswith("mp3_"):
            rmses=[]; sample_deltas=[]
            for item in manifest[:20]:
                original=read_audio(eval_root/f"{item['utterance']}.flac",sr)
                decoded=read_audio(WAVE_ROOT/condition/f"{item['utterance']}.wav",sr)
                sample_deltas.append(int(len(decoded)-len(original)))
                common=min(len(original),len(decoded))
                delta=(decoded[:common]-original[:common]).astype(np.float64)
                rmses.append(float(np.sqrt(np.mean(delta**2))))
            mp3_roundtrip[condition]={"clips":len(rmses),"sample_count_delta_min":min(sample_deltas),
                                      "sample_count_delta_max":max(sample_deltas),
                                      "clips_with_length_change":sum(d!=0 for d in sample_deltas),
                                      "mean_common_prefix_error_rms":float(np.mean(rmses)),
                                      "max_common_prefix_error_rms":float(max(rmses))}
    validation_report={"feature_waveform_validation_clips_per_condition":20,"batch_inference_validation_clips_per_condition":32,"reference_batch_size":32,"inference_equivalence":inference_equivalence,
                       "mp3_roundtrip":mp3_roundtrip,"ffmpeg_version":ffmpeg_version,"mp3_encoder":"libmp3lame",
                       "sample_rate":sr,"score_tolerance":1e-6,
                       "waveform_feature_max_abs_diff_by_condition":{
                           c:result["conditions"][c]["runtime"]["optimization_equivalence"] for c in conditions}}
    validation_path=RUN_OUT/"phase4b_optimization_validation.json"
    if validation_path.exists():
        previous=json.loads(validation_path.read_text(encoding="utf-8"))
        if previous!=validation_report:
            raise ValueError("Existing optimization validation report differs; preserve and investigate")
    else:
        validation_path.write_text(json.dumps(validation_report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    aggregate_path=RUN_OUT/"phase4b_metrics.json"
    if aggregate_path.exists():
        previous=json.loads(aggregate_path.read_text(encoding="utf-8"))
        if (previous.get("status")!="completed" or previous.get("manifest_sha256")!=result["manifest_sha256"]
                or previous.get("conditions")!=result["conditions"] or previous.get("clean_reference")!=result["clean_reference"]):
            raise ValueError("Existing aggregate metrics differ from validated saved outputs; preserve and investigate")
        result=previous
    else:
        plot_curves(result)
        aggregate_path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    summary_path=RUN_OUT/"phase4b_summary.md"
    destination=(RUN_OUT/"phase4b_summary_verified.md") if summary_path.exists() else summary_path
    write_summary(result,destination)
    print(json.dumps({"status":result["status"],"finalized_from_saved_scores":True,
                      "conditions":conditions,"output":str(RUN_OUT)},indent=2),flush=True)
    return result


def write_blocked_summary_from_saved_debug():
    timing=json.loads((OUT/"debug_timing.json").read_text(encoding="utf-8"))
    if timing.get("status")!="blocked_runtime_over_3h":
        raise ValueError("Saved timing evidence does not show a blocked runtime gate")
    storage=timing["free_disk_preflight"]
    lines=["# Phase 4B preflight: BLOCKED by runtime gate", "",
           "DEBUG ONLY timing on the fixed 10,000-clip subset. No full-subset robustness metrics were computed.", "",
           f"- Manifest SHA-256: `{sha256(MANIFEST)}`; subset N=10,000, 1,030 bona fide, 8,970 spoof (690 per A07–A19).",
           f"- 500-clip MP3-64 debug wall time: {timing['measured_seconds']['wall_seconds']:.3f}s, including waveform encoding/decoding, disk writes, both feature caches, feature reads, and both model inferences.",
           f"- Conservative total estimate for six conditions and both models: {timing['total_six_conditions_both_models_estimate_seconds']:.3f}s ({timing['total_six_conditions_both_models_estimate_seconds']/3600:.3f}h), above the 10,800s gate.",
           f"- Disk preflight: {storage['free_disk_bytes_before']:,} bytes free; estimated waveform/cache need {storage['estimated_total_bytes']:,} bytes; estimated remaining free {storage['estimated_free_after_bytes']:,} bytes.",
           f"- FFmpeg: `{timing['ffmpeg_version']}` via imageio-ffmpeg fallback.",
           "- Full robustness execution stopped as required; the fixed subset was not reduced or resampled.", "",
           "Options: approve a runtime budget above the estimate; approve a new, smaller fixed subset; or request an implementation/timing optimization that retains all 10,000 clips, followed by a fresh 500-clip preflight. No option is applied automatically.", ""]
    (OUT/"phase4b_summary.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-only",action="store_true",help="time 500 fixed-subset MP3-64 clips and enforce the three-hour runtime gate")
    parser.add_argument("--write-blocked-summary-only",action="store_true",help="generate the blocked preflight report from saved debug timing without running inference")
    parser.add_argument("--finalize-saved",action="store_true",help="validate completed condition score files and build aggregate metrics/plots without rerunning inference")
    parser.add_argument("--write-summary-from-saved-only",action="store_true",help="write the report from already validated aggregate metrics and optimization validation files")
    args=parser.parse_args()
    if args.write_blocked_summary_only:
        write_blocked_summary_from_saved_debug()
        return
    cnn_meta=json.loads((CNN_DIR/"training.json").read_text(encoding="utf-8"))
    svm_meta=json.loads((SVM_REFERENCE/"training.json").read_text(encoding="utf-8"))
    if args.write_summary_from_saved_only:
        result=json.loads((RUN_OUT/"phase4b_metrics.json").read_text(encoding="utf-8"))
        if result.get("status")!="completed" or not (RUN_OUT/"phase4b_optimization_validation.json").exists():
            raise ValueError("Validated aggregate metrics and optimization validation must exist")
        write_summary(result)
        return
    if args.finalize_saved:
        finalize_saved_results(load_rows(MANIFEST),cnn_meta,svm_meta)
        return
    cfg=cnn_meta["config"]
    data,_=load_dataset(cfg)
    manifest=load_rows(MANIFEST)
    row_map={r["utterance"]:r for r in data["eval"]}
    if len(manifest)!=N_SUBSET or len({r['utterance'] for r in manifest})!=N_SUBSET:
        raise ValueError("Subset manifest must contain 10,000 unique trials")
    rows=[]
    for item in manifest:
        row=row_map.get(item["utterance"])
        if row is None or any(row[key]!=item[key] for key in ("speaker","label","attack")):
            raise ValueError(f"Subset manifest/protocol mismatch: {item['utterance']}")
        rows.append(row)
    if sum(r["label"]==0 for r in rows)!=1030 or sum(r["label"]==1 for r in rows)!=8970:
        raise ValueError("Unexpected fixed subset class counts")
    if sha256(MANIFEST)!=json.loads((OUT/"subset_manifest_metadata.json").read_text(encoding="utf-8"))["manifest_sha256"]:
        raise ValueError("Subset manifest hash mismatch")
    if not args.debug_only and (RUN_OUT/"phase4b_metrics.json").exists():
        raise FileExistsError("Refusing to overwrite existing Phase 4B metrics")
    cnn_model,cnn_scaler,cnn_device=load_model(CNN_DIR,cnn_meta)
    svm_model,_,_=load_model(SVM_ARTIFACTS,svm_meta)
    svm_cfg=svm_meta["config"]
    free_before=shutil.disk_usage(ROOT).free
    storage=estimate_storage(rows,cfg,svm_cfg,free_before)
    if storage["estimated_free_after_bytes"]<20*1024**3:
        raise RuntimeError("Disk preflight leaves less than 20 GiB after the conservative estimate")
    if args.debug_only:
        result=run_debug(rows,cfg,svm_cfg,(cnn_model,cnn_scaler,cnn_device,svm_model),free_before)
        if result["status"]!="pass":
            raise SystemExit(2)
        return
    timing_path=OUT/"optimized_debug_timing.json"
    if not timing_path.exists(): raise RuntimeError("Run the inspected --debug-only preflight before full robustness work")
    debug=json.loads(timing_path.read_text(encoding="utf-8"))
    if debug["status"]!="pass" or debug["total_six_conditions_both_models_estimate_seconds"]>MAX_RUNTIME_SECONDS:
        raise RuntimeError("Three-hour debug timing gate did not pass")
    RUN_OUT.mkdir(parents=True, exist_ok=True)
    conditions=["noise_20dB","noise_10dB","noise_0dB","mp3_128kbps","mp3_64kbps","trimmed"]
    reference_scores=read_clean_reference(rows)
    thresholds={"CNN":float(cnn_meta["threshold_selection"]["threshold"]),
                "SVM":float(svm_meta["threshold_selection"]["threshold"])}
    result={"status":"running","scope":"SUBSET RESULTS only","manifest_sha256":sha256(MANIFEST),
            "subset":{"n":len(rows),"bona_fide":1030,"spoof":8970,"seed":2026,"sha256":sha256(MANIFEST)},
            "thresholds_full_precision":{k:repr(v) for k,v in thresholds.items()},
            "debug_preflight":debug,"storage_preflight":storage,"clean_reference":{},"conditions":{}}
    labels=[r["label"] for r in rows]
    for model in ("CNN","SVM"):
        result["clean_reference"][model]=metric_report(rows,reference_scores[model],thresholds[model])
    all_scores={}
    for condition in conditions:
        if (RUN_OUT/f"{condition}_metrics.json").exists():
            raise FileExistsError(f"Refusing to overwrite condition evidence: {condition}")
        cnn_scores,svm_scores,times=process_condition(rows,condition,cfg,cnn_model,cnn_scaler,cnn_device,svm_model,svm_cfg)
        all_scores[condition]={"CNN":cnn_scores,"SVM":svm_scores}
        result["conditions"][condition]={
            "CNN":metric_report(rows,cnn_scores,thresholds["CNN"]),
            "SVM":metric_report(rows,svm_scores,thresholds["SVM"]),
            "runtime":times}
        for model,scores in (("CNN",cnn_scores),("SVM",svm_scores)):
            csv_path=RUN_OUT/f"{condition}_{model.lower()}_scores.csv"
            with csv_path.open("w",newline="",encoding="utf-8") as f:
                writer=csv.writer(f); writer.writerow(["utterance","speaker","label","attack","score"])
                writer.writerows([[r["utterance"],r["speaker"],r["label"],r["attack"],format(float(score),".17g")] for r,score in zip(rows,scores)])
        (RUN_OUT/f"{condition}_metrics.json").write_text(json.dumps(result["conditions"][condition],indent=2,allow_nan=False)+"\n",encoding="utf-8")
        (RUN_OUT/"phase4b_progress.json").write_text(json.dumps({"completed_conditions":list(result["conditions"]),"manifest_sha256":result["manifest_sha256"]},indent=2)+"\n",encoding="utf-8")
    result["status"]="completed"
    plot_curves(result)
    (RUN_OUT/"phase4b_metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    write_summary(result)
    print(json.dumps({"status":result["status"],"conditions":conditions,"output":str(RUN_OUT)},indent=2),flush=True)


if __name__=="__main__": main()
