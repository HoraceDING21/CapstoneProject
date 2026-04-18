from __future__ import annotations

import os

from flask import Flask, abort, render_template, request, send_file

from bev_demo.core import DemoRunResult, SampleEntry, get_runtime


app = Flask(__name__, template_folder="templates", static_folder="static")


def _resolve_selected_sample(sample_id: str | None) -> SampleEntry | None:
    runtime = get_runtime()
    return runtime.catalog.get(sample_id) or runtime.catalog.first()


@app.route("/", methods=["GET", "POST"])
def index():
    runtime = get_runtime()
    selected_sample = _resolve_selected_sample(request.values.get("sample_id"))
    result: DemoRunResult | None = None
    error: str | None = None
    pitch_image_data_uri = runtime.service.get_empty_pitch_image_data_uri()

    if request.method == "POST":
        if selected_sample is None:
            error = "No sample image is available. Update the local demo config first."
        else:
            try:
                result = runtime.service.run(selected_sample)
                pitch_image_data_uri = result.pitch_image_data_uri
            except Exception as exc:  # pragma: no cover - exercised manually in the local demo
                error = str(exc)

    return render_template(
        "index.html",
        title=runtime.config.title,
        samples=runtime.catalog.samples,
        selected_sample=selected_sample,
        notices=runtime.catalog.errors,
        run_error=error,
        result=result,
        pitch_image_data_uri=pitch_image_data_uri,
    )


@app.route("/sample-image/<sample_id>")
def sample_image(sample_id: str):
    sample = _resolve_selected_sample(sample_id)
    if sample is None or sample.id != sample_id:
        abort(404)
    return send_file(sample.image_path)


@app.route("/healthz")
def healthz():
    runtime = get_runtime()
    return {
        "status": "ok",
        "sample_count": len(runtime.catalog.samples),
        "config_path": str(runtime.config.config_path) if runtime.config.config_path else None,
    }


if __name__ == "__main__":
    app.run(
        host=os.getenv("BEV_DEMO_HOST", "127.0.0.1"),
        port=int(os.getenv("BEV_DEMO_PORT", "5000")),
        debug=os.getenv("BEV_DEMO_DEBUG", "1") != "0",
    )
