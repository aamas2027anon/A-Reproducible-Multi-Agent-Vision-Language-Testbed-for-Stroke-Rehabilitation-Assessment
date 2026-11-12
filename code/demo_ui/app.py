"""Gradio controls and result presentation for the MAS demo."""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from demo_ui.outputs import PREDICTION_COLUMNS, inspect_window, load_output
from demo_ui.runner import DemoOptions, config, input_defaults, run_demo


def _path(value: str, label: str, *, required: bool = True) -> Path | None:
    value = (value or "").strip()
    if not value:
        if required:
            raise ValueError(f"Enter {label}.")
        return None
    return Path(value).expanduser().absolute()


def _integer(value, label: str) -> int:
    if value is None or int(value) != value:
        raise ValueError(f"{label} must be a whole number.")
    return int(value)


def build_app() -> gr.Blocks:
    checkpoint_default, settings_default, default_notes = input_defaults()
    with gr.Blocks(title="Stroke rehabilitation demo", analytics_enabled=False) as app:
        gr.Markdown("# Stroke rehabilitation demo\nChoose a clip and patient history, run the assessment, and inspect the decisions.")
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                sample = gr.Textbox(value="s04_drinking3_1", label="Clip ID")
                checkpoint = gr.Textbox(value=checkpoint_default, label="Model checkpoint (.ckpt)")
                tuned = gr.Textbox(value=settings_default, label="MAS settings (.json)",
                                   info="Use settings tuned for this checkpoint. Blank uses config.yaml agent settings.")
                if default_notes:
                    with gr.Accordion("Checkpoint selection", open=True):
                        gr.Textbox(value=default_notes, show_label=False, interactive=False)
                history = gr.Radio(["None", "Last N clips", "All preceding clips"],
                                   value="None", label="Prior history")
                history_count = gr.Number(value=5, precision=0, minimum=0,
                                          label="Number of preceding clips", visible=False)
                memory = gr.Radio(["Fresh for this run", "Continue persistent memory"],
                                  value="Fresh for this run", label="Patient memory")
                persistent = gr.Textbox(
                    value=str(config.DEMO_OUTPUT_DIR / "ui" / "patient_memory.sqlite"),
                    label="Persistent SQLite file", visible=False,
                    info="An existing patient record is continued. History selection does not clear it.",
                )
                with gr.Accordion("Advanced options", open=False):
                    raw = gr.Textbox(value=str(config.RAW_DATA_DIR), label="Raw data directory")
                    base = gr.Textbox(value=str(config.BASE_MODEL_DIR), label="Base model directory")
                    device = gr.Textbox(value="auto", label="Device", info="auto, cuda, cuda:0 or cpu")
                    batch = gr.Number(value=16, precision=0, minimum=1, label="Model batch size")
                    fast = gr.Checkbox(value=True, label="Fast streaming (no pacing delays)")
                    download = gr.Checkbox(value=True, label="Allow base-model download if missing")
                    output_root = gr.Textbox(value=str(config.DEMO_OUTPUT_DIR / "ui"),
                                             label="Output parent directory")
                run = gr.Button("Run demo", variant="primary")
                status = gr.Textbox(value="Ready.", label="Execution status", interactive=False, lines=2)
            with gr.Column(scale=3, min_width=500):
                summary = gr.Markdown("Run a clip to see its assessment.")
                notes = gr.Markdown()
                with gr.Tabs():
                    with gr.Tab("Assessment"):
                        timeline = gr.Image(label="Final output and status · frame index", type="pil", interactive=False, height=220)
                        gr.Markdown("Hatched final predictions were withheld.")
                        clinician = gr.Textbox(label="Clinician report", lines=7, interactive=False)
                    with gr.Tab("Decision inspector"):
                        window = gr.Dropdown(choices=[], label="Window / frame range", interactive=False)
                        predictions = gr.Dataframe(headers=PREDICTION_COLUMNS, value=[], interactive=False,
                                                   label="Predictions for this window")
                        decision = gr.Textbox(label="Decision and safety details", lines=10, interactive=False)
                    with gr.Tab("Patient trajectory"):
                        trajectory = gr.Image(label="Patient history", type="pil", interactive=False)
                    with gr.Tab("Technical details"):
                        with gr.Accordion("Technical report", open=False):
                            technical = gr.Textbox(show_label=False, lines=15, interactive=False)
                with gr.Accordion("Execution log / progress", open=False):
                    log = gr.Textbox(show_label=False, lines=15, max_lines=20, interactive=False)
        windows = gr.State([])

        history.change(
            lambda choice: gr.Number(visible=choice == "Last N clips"),
            inputs=history, outputs=history_count, queue=False,
        )
        memory.change(
            lambda choice: gr.Textbox(visible=choice == "Continue persistent memory"),
            inputs=memory, outputs=persistent, queue=False,
        )
        window.input(inspect_window, inputs=[windows, window], outputs=[predictions, decision])

        def execute(sample_value, checkpoint_value, tuned_value, history_value, count_value,
                    memory_value, persistent_value, raw_value, base_value, device_value,
                    batch_value, fast_value, download_value, output_value):
            # Every run starts with an empty result pane, including failed requests.
            yield {
                run: gr.Button(interactive=False), status: "Preparing demo…", log: "",
                summary: "Assessment running…", notes: "", timeline: None, trajectory: None,
                clinician: "", technical: "", windows: [],
                window: gr.Dropdown(choices=[], value=None, interactive=False),
                predictions: [], decision: "",
            }
            try:
                options = DemoOptions(
                    sample_id=(sample_value or "").strip(),
                    checkpoint=_path(checkpoint_value, "a checkpoint path"),
                    mas_config=_path(tuned_value, "MAS settings", required=False),
                    history_mode=history_value,
                    history_clips=_integer(count_value, "History count") if history_value == "Last N clips" else 0,
                    fresh_memory=memory_value == "Fresh for this run",
                    persistent_memory=_path(persistent_value, "persistent memory", required=False),
                    raw_dir=_path(raw_value, "the raw data directory"),
                    base_model=_path(base_value, "the base model directory", required=False),
                    device=(device_value or "auto").strip(),
                    batch_size=_integer(batch_value, "Batch size"),
                    fast=bool(fast_value), allow_download=bool(download_value),
                    output_root=_path(output_value, "an output parent directory"),
                )
                for update in run_demo(options):
                    values = {status: update.status, log: update.log}
                    if update.complete:
                        values[run] = gr.Button(interactive=True)
                        if update.succeeded:
                            try:
                                result = load_output(update.run_dir / options.sample_id, options.sample_id)
                                choices = [
                                    (f"Window {row['window_index']} · frame {row['target_frame_index']}",
                                     str(row["window_index"]))
                                    for row in result.windows
                                ]
                                first = choices[0][1] if choices else None
                                table, detail = inspect_window(result.windows, first)
                                values.update({
                                    status: f"Completed. Outputs: {update.run_dir}",
                                    summary: result.summary, notes: result.notes,
                                    timeline: result.timeline, trajectory: result.trajectory,
                                    clinician: result.clinician, technical: result.technical,
                                    windows: result.windows, predictions: table, decision: detail,
                                    window: gr.Dropdown(choices=choices, value=first, interactive=bool(choices)),
                                })
                            except (OSError, ValueError, KeyError) as error:
                                values.update({status: "Demo completed, but its exports could not be displayed.",
                                               summary: "Results unavailable.",
                                               notes: f"{error}\n\nOutputs remain at `{update.run_dir}`."})
                        else:
                            values[summary] = "No results from this request. See execution status and log."
                    yield values
            except Exception as error:
                yield {run: gr.Button(interactive=True), status: f"Failed: {error}",
                       summary: "No results from this request."}

        run.click(
            execute,
            inputs=[sample, checkpoint, tuned, history, history_count, memory, persistent,
                    raw, base, device, batch, fast, download, output_root],
            outputs=[run, status, log, summary, notes, timeline, trajectory, clinician,
                     technical, windows, window, predictions, decision],
            concurrency_limit=None, trigger_mode="once",
        )
    return app
