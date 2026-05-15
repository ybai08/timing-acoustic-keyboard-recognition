from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from keyboard_fusion.paths import MODELS_DIR


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def find_latest_acoustic_baseline_session(model_root: Path | None = None) -> Path:
    root = model_root or MODELS_DIR / "acoustic_baseline"
    sessions = sorted(path for path in root.iterdir() if path.is_dir())
    if not sessions:
        raise FileNotFoundError(f"No acoustic baseline sessions found under {root}")
    return sessions[-1]


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def rounded_grid(values: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 5) for value in row] for row in values]


def weight_heatmaps(model: Any, feature_shape: list[int]) -> list[dict[str, Any]]:
    classifier = model.named_steps["classifier"]
    classes = [str(label) for label in classifier.classes_]
    coefficients = np.asarray(classifier.coef_, dtype=np.float32)
    shape = tuple(feature_shape) if feature_shape else (int(coefficients.shape[1]),)
    heatmaps: list[dict[str, Any]] = []

    for index, key in enumerate(classes):
        if index >= coefficients.shape[0]:
            break
        if len(shape) == 1:
            weights = coefficients[index].reshape(1, shape[0])
        else:
            weights = coefficients[index].reshape(shape)
        heatmaps.append(
            {
                "key": key,
                "minimum": round(float(np.min(weights)), 6),
                "maximum": round(float(np.max(weights)), 6),
                "weights": rounded_grid(weights),
            }
        )
    return heatmaps


def prediction_payload(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row in rows:
        top: list[dict[str, Any]] = []
        for rank in range(1, 6):
            key = row.get(f"top{rank}_key", "")
            if not key:
                continue
            top.append(
                {
                    "rank": rank,
                    "key": key,
                    "probability": parse_float(row.get(f"top{rank}_probability")),
                }
            )
        predictions.append(
            {
                "clipId": row.get("clip_id", ""),
                "trialId": row.get("trial_id", ""),
                "eventIndex": row.get("event_index", ""),
                "trueKey": row.get("true_key", ""),
                "predictedKey": row.get("predicted_key", ""),
                "correctTop1": parse_int(row.get("correct_top1")),
                "trueInTop5": parse_int(row.get("true_in_top5")),
                "top": top,
            }
        )
    return predictions


def confusion_payload(predictions: list[dict[str, Any]], classes: list[str]) -> dict[str, Any]:
    labels = sorted(set(classes) | {row["trueKey"] for row in predictions} | {row["predictedKey"] for row in predictions})
    index_by_label = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for row in predictions:
        true_index = index_by_label[row["trueKey"]]
        predicted_index = index_by_label[row["predictedKey"]]
        matrix[true_index][predicted_index] += 1
    return {"labels": labels, "matrix": matrix}


def build_visualization_payload(model_dir: Path) -> dict[str, Any]:
    model = joblib.load(model_dir / "model.joblib")
    metrics = read_json(model_dir / "metrics.json")
    prediction_rows = read_csv(model_dir / "test_predictions.csv")

    classifier = model.named_steps["classifier"]
    scaler = model.named_steps["scale"]
    classes = [str(label) for label in classifier.classes_]
    feature_shape = [int(value) for value in metrics.get("feature_shape", [])]
    input_features = int(np.prod(feature_shape)) if feature_shape else int(classifier.coef_.shape[1])
    if not feature_shape:
        feature_shape = [input_features]
    trainable_parameters = int(classifier.coef_.size + classifier.intercept_.size)
    predictions = prediction_payload(prediction_rows)

    return {
        "architecture": {
            "sessionId": metrics.get("session_id", model_dir.name),
            "modelType": metrics.get("model_type", "logistic_regression_flattened_log_mel"),
            "isNeuralNetwork": False,
            "inputShape": feature_shape,
            "inputFeatures": input_features,
            "hiddenLayers": 0,
            "hiddenNeurons": 0,
            "outputClasses": len(classes),
            "weightMatrixShape": [int(value) for value in classifier.coef_.shape],
            "interceptShape": [int(value) for value in classifier.intercept_.shape],
            "trainableParameters": trainable_parameters,
            "pipeline": [
                {"name": "Log-mel spectrogram", "detail": " x ".join(str(value) for value in feature_shape)},
                {"name": "Flatten", "detail": f"{input_features} numeric input features"},
                {"name": "StandardScaler", "detail": f"Normalize {int(getattr(scaler, 'n_features_in_', input_features))} features"},
                {"name": "LogisticRegression", "detail": f"{len(classes)} key probability outputs"},
            ],
        },
        "metrics": metrics,
        "classes": classes,
        "weightHeatmaps": weight_heatmaps(model, feature_shape),
        "predictions": predictions,
        "confusion": confusion_payload(predictions, classes),
    }


def html_template() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Acoustic Model Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #090b10;
      --panel: #111722;
      --panel-strong: #151d2b;
      --line: #263244;
      --soft-line: rgba(154, 168, 188, 0.18);
      --text: #edf3ff;
      --muted: #9aa8bc;
      --blue: #49a5ff;
      --green: #68d391;
      --amber: #ffcc66;
      --red: #ff7a7a;
    }
    * { box-sizing: border-box; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      min-width: 320px;
    }
    .shell {
      display: grid;
      gap: 18px;
      padding: 24px;
    }
    header {
      align-items: end;
      display: flex;
      gap: 16px;
      justify-content: space-between;
    }
    h1 {
      font-size: clamp(24px, 3vw, 34px);
      line-height: 1.05;
      margin: 0;
    }
    .summary {
      color: var(--muted);
      font-size: 14px;
      margin: 8px 0 0;
    }
    .grid {
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }
    .stat, .panel, .model-step, .prediction {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .stat {
      min-height: 90px;
      padding: 14px;
    }
    .stat strong {
      display: block;
      font-size: 28px;
      line-height: 1;
    }
    .stat span {
      color: var(--muted);
      display: block;
      font-size: 12px;
      margin-top: 8px;
    }
    .panel {
      padding: 14px;
    }
    .panel h2 {
      font-size: 16px;
      margin: 0 0 12px;
    }
    .architecture {
      align-items: stretch;
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
    }
    .model-step {
      padding: 13px;
      position: relative;
    }
    .model-step strong {
      display: block;
      font-size: 14px;
      margin-bottom: 6px;
    }
    .model-step span {
      color: var(--muted);
      display: block;
      font-size: 12px;
      line-height: 1.35;
    }
    .controls {
      align-items: center;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      justify-content: space-between;
      margin-bottom: 10px;
    }
    label {
      color: var(--muted);
      display: grid;
      font-size: 12px;
      font-weight: 700;
      gap: 6px;
      text-transform: uppercase;
    }
    select, input {
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--text);
      font: inherit;
      min-height: 40px;
      padding: 10px 12px;
    }
    canvas {
      background: #05070b;
      border: 1px solid var(--soft-line);
      display: block;
      height: 270px;
      width: 100%;
    }
    .heatmap-meta {
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      font-size: 12px;
      gap: 10px;
      justify-content: space-between;
      margin-top: 8px;
    }
    .two-column {
      display: grid;
      gap: 14px;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    }
    table {
      border-collapse: collapse;
      font-size: 12px;
      width: 100%;
    }
    th, td {
      border-bottom: 1px solid var(--soft-line);
      padding: 7px;
      text-align: right;
    }
    th:first-child, td:first-child { text-align: left; }
    th {
      color: var(--muted);
      font-weight: 700;
    }
    .prediction-list {
      display: grid;
      gap: 8px;
      max-height: 520px;
      overflow: auto;
      padding-right: 4px;
    }
    .prediction {
      display: grid;
      gap: 8px;
      padding: 10px;
    }
    .prediction-title {
      align-items: center;
      display: flex;
      gap: 8px;
      justify-content: space-between;
    }
    .prediction-title strong {
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .pill {
      border: 1px solid var(--soft-line);
      border-radius: 6px;
      color: var(--muted);
      font-size: 11px;
      padding: 4px 7px;
      white-space: nowrap;
    }
    .pill.good {
      color: var(--green);
    }
    .pill.bad {
      color: var(--red);
    }
    .bars {
      display: grid;
      gap: 5px;
    }
    .bar-row {
      align-items: center;
      display: grid;
      gap: 8px;
      grid-template-columns: 34px minmax(0, 1fr) 50px;
    }
    .bar-track {
      background: #070a0f;
      border: 1px solid var(--soft-line);
      border-radius: 999px;
      height: 9px;
      overflow: hidden;
    }
    .bar-fill {
      background: linear-gradient(90deg, var(--blue), var(--green));
      height: 100%;
    }
    .muted {
      color: var(--muted);
    }
    @media (max-width: 900px) {
      .shell { padding: 16px; }
      header { align-items: stretch; flex-direction: column; }
      .architecture, .two-column { grid-template-columns: 1fr; }
      canvas { height: 220px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>Acoustic Model Viewer</h1>
        <p class="summary" id="summary"></p>
      </div>
    </header>

    <section class="grid" id="stats"></section>

    <section class="panel">
      <h2>Model Structure</h2>
      <div class="architecture" id="architecture"></div>
    </section>

    <section class="two-column">
      <div class="panel">
        <div class="controls">
          <h2>Per-Key Weight Heatmap</h2>
          <label for="keySelect">
            Key
            <select id="keySelect"></select>
          </label>
        </div>
        <canvas id="heatmap"></canvas>
        <div class="heatmap-meta" id="heatmapMeta"></div>
      </div>

      <div class="panel">
        <h2>Confusion Matrix</h2>
        <div id="confusion"></div>
      </div>
    </section>

    <section class="panel">
      <div class="controls">
        <h2>Held-Out Predictions</h2>
        <label for="predictionFilter">
          Filter
          <select id="predictionFilter">
            <option value="all">All predictions</option>
            <option value="wrong">Wrong top-1 only</option>
            <option value="top5miss">Not in top-5 only</option>
          </select>
        </label>
      </div>
      <div class="prediction-list" id="predictionList"></div>
    </section>
  </div>

  <script id="visualization-data" type="application/json">__PAYLOAD__</script>
  <script>
    const data = JSON.parse(document.getElementById("visualization-data").textContent);
    const summary = document.getElementById("summary");
    const stats = document.getElementById("stats");
    const architecture = document.getElementById("architecture");
    const keySelect = document.getElementById("keySelect");
    const heatmap = document.getElementById("heatmap");
    const heatmapMeta = document.getElementById("heatmapMeta");
    const confusion = document.getElementById("confusion");
    const predictionFilter = document.getElementById("predictionFilter");
    const predictionList = document.getElementById("predictionList");

    function addStat(value, label) {
      const card = document.createElement("div");
      card.className = "stat";
      const strong = document.createElement("strong");
      strong.textContent = value;
      const span = document.createElement("span");
      span.textContent = label;
      card.append(strong, span);
      stats.appendChild(card);
    }

    function renderStats() {
      const a = data.architecture;
      const m = data.metrics;
      summary.textContent = `${a.sessionId} | current baseline: logistic regression, not a neural network`;
      addStat(a.inputFeatures, "input features from the spectrogram");
      addStat(a.hiddenNeurons, "hidden neurons in this baseline");
      addStat(a.outputClasses, "key classes predicted");
      addStat(a.trainableParameters.toLocaleString(), "trainable weights + intercepts");
      addStat(Number(m.top1_accuracy).toFixed(3), "top-1 held-out accuracy");
      addStat(Number(m.top5_accuracy).toFixed(3), "top-5 held-out accuracy");
    }

    function renderArchitecture() {
      data.architecture.pipeline.forEach((step) => {
        const box = document.createElement("div");
        box.className = "model-step";
        const title = document.createElement("strong");
        title.textContent = step.name;
        const detail = document.createElement("span");
        detail.textContent = step.detail;
        box.append(title, detail);
        architecture.appendChild(box);
      });
    }

    function heatColor(value, maxAbs) {
      if (!maxAbs) return "hsl(210 18% 10%)";
      const intensity = Math.min(1, Math.abs(value) / maxAbs);
      if (value >= 0) {
        return `hsl(188 82% ${12 + intensity * 66}%)`;
      }
      return `hsl(28 92% ${13 + intensity * 56}%)`;
    }

    function setupCanvas(canvas) {
      const width = Math.max(320, Math.floor(canvas.clientWidth || 320));
      const height = Math.max(220, Math.floor(canvas.clientHeight || 220));
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      return { ctx, width, height };
    }

    function drawHeatmap(index) {
      const item = data.weightHeatmaps[index];
      const { ctx, width, height } = setupCanvas(heatmap);
      const rows = item.weights.length;
      const cols = item.weights[0].length;
      const maxAbs = Math.max(Math.abs(item.minimum), Math.abs(item.maximum));
      const cellWidth = width / cols;
      const cellHeight = height / rows;
      for (let row = 0; row < rows; row += 1) {
        for (let col = 0; col < cols; col += 1) {
          ctx.fillStyle = heatColor(item.weights[row][col], maxAbs);
          ctx.fillRect(col * cellWidth, height - (row + 1) * cellHeight, cellWidth + 0.5, cellHeight + 0.5);
        }
      }
      heatmapMeta.textContent = `Key ${item.key}: positive cyan cells push the model toward this key; amber cells push away. Weight range ${item.minimum} to ${item.maximum}.`;
    }

    function renderKeySelect() {
      data.weightHeatmaps.forEach((item, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = item.key;
        keySelect.appendChild(option);
      });
      keySelect.addEventListener("change", () => drawHeatmap(Number(keySelect.value)));
      drawHeatmap(0);
    }

    function renderConfusion() {
      const labels = data.confusion.labels;
      const matrix = data.confusion.matrix;
      const table = document.createElement("table");
      const head = document.createElement("thead");
      const headRow = document.createElement("tr");
      ["true \\ predicted", ...labels].forEach((label) => {
        const th = document.createElement("th");
        th.textContent = label;
        headRow.appendChild(th);
      });
      head.appendChild(headRow);
      table.appendChild(head);

      const body = document.createElement("tbody");
      matrix.forEach((row, rowIndex) => {
        const tr = document.createElement("tr");
        const label = document.createElement("td");
        label.textContent = labels[rowIndex];
        tr.appendChild(label);
        row.forEach((value, colIndex) => {
          const td = document.createElement("td");
          td.textContent = value || "";
          if (rowIndex === colIndex && value) td.style.color = "var(--green)";
          tr.appendChild(td);
        });
        body.appendChild(tr);
      });
      table.appendChild(body);
      confusion.appendChild(table);
    }

    function predictionMatchesFilter(row, filter) {
      if (filter === "wrong") return !row.correctTop1;
      if (filter === "top5miss") return !row.trueInTop5;
      return true;
    }

    function renderPredictions() {
      const filter = predictionFilter.value;
      predictionList.textContent = "";
      data.predictions.filter((row) => predictionMatchesFilter(row, filter)).forEach((row) => {
        const card = document.createElement("article");
        card.className = "prediction";

        const title = document.createElement("div");
        title.className = "prediction-title";
        const strong = document.createElement("strong");
        strong.textContent = row.clipId;
        const pill = document.createElement("span");
        pill.className = `pill ${row.correctTop1 ? "good" : "bad"}`;
        pill.textContent = `true ${row.trueKey} | predicted ${row.predictedKey}`;
        title.append(strong, pill);
        card.appendChild(title);

        const bars = document.createElement("div");
        bars.className = "bars";
        row.top.forEach((candidate) => {
          const line = document.createElement("div");
          line.className = "bar-row";
          const key = document.createElement("span");
          key.textContent = candidate.key;
          const track = document.createElement("div");
          track.className = "bar-track";
          const fill = document.createElement("div");
          fill.className = "bar-fill";
          fill.style.width = `${Math.max(0, Math.min(1, candidate.probability)) * 100}%`;
          track.appendChild(fill);
          const value = document.createElement("span");
          value.className = "muted";
          value.textContent = candidate.probability.toFixed(3);
          line.append(key, track, value);
          bars.appendChild(line);
        });
        card.appendChild(bars);
        predictionList.appendChild(card);
      });
    }

    let resizeTimer = null;
    window.addEventListener("resize", () => {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(() => drawHeatmap(Number(keySelect.value || 0)), 120);
    });
    predictionFilter.addEventListener("change", renderPredictions);

    renderStats();
    renderArchitecture();
    renderKeySelect();
    renderConfusion();
    renderPredictions();
  </script>
</body>
</html>
"""


def build_acoustic_visualization_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))
    payload_json = (
        payload_json.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    return html_template().replace("__PAYLOAD__", payload_json)


def generate_acoustic_visualization(model_dir: Path, output_path: Path | None = None) -> Path:
    payload = build_visualization_payload(model_dir)
    path = output_path or model_dir / "model_visualization.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_acoustic_visualization_html(payload), encoding="utf-8")
    return path
