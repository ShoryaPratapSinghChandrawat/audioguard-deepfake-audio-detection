"""Local Gradio research demo using a frozen run and JSON-backed evaluation panel.

The displayed benchmark numbers are never literals: absent results show an explicit
unavailable message. Arbitrary audio is not a validated deployment distribution.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gradio as gr
from scripts.experiment import load_model, predict_rows

NOTICE = """Research prototype — not a forensic verdict or calibrated confidence.
Synthetic/unknown-domain audio, languages, microphones, silence and compression can
produce errors. A detector cannot establish a speaker's identity or consent.
Upload mono audio at the sample rate specified in the frozen frontend configuration.
Evaluation statistics below come only from this run's saved results, not this upload.
Do not upload sensitive recordings to a shared instance. No public share link is enabled.
"""


def build(run):
    run = Path(run)
    meta = json.loads((run / "training.json").read_text(encoding="utf-8"))
    model, scaler, device = load_model(run, meta)
    result_path = run / "eval_clean.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        panel = {"scope": meta["scope"], "status": result["status"],
                 "metrics": result.get("metrics"), "data_scope": result.get("dataset_audit", {}).get("used_counts")}
    else:
        panel = {"status": "No saved evaluation results. Performance is unknown.", "scope": meta["scope"]}

    def predict(path):
        if path is None:
            raise gr.Error("Choose an audio file first.")
        try:
            row = {"path": path, "label": 0, "utterance": "user_upload"}
            value = float(predict_rows([row], meta["config"], meta["name"], model, scaler, device)[0])
            threshold = meta["threshold_selection"]["threshold"]
            return {"score": value, "dev_threshold": threshold,
                    "prediction": "spoof" if value >= threshold else "bonafide",
                    "score_direction": "higher means spoof; not calibrated confidence"}
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="AudioGuard research demo") as app:
        gr.Markdown("# AudioGuard\n" + NOTICE)
        gr.JSON(panel, label="Saved evaluation evidence")
        audio = gr.Audio(type="filepath", label="Upload mono audio", sources=["upload"])
        button = gr.Button("Analyze with frozen dev threshold")
        output = gr.JSON(label="Research prediction")
        button.click(predict, inputs=audio, outputs=output)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="results/research/cnn_logmel")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build(args.run).launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
