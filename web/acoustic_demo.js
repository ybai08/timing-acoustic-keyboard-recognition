const state = {
  config: {},
  model: {},
  recording: false,
  audioContext: null,
  mediaStream: null,
  sourceNode: null,
  processorNode: null,
  muteNode: null,
  audioChunks: [],
  audioDevices: [],
  selectedAudioDeviceId: "",
  sampleRate: 48000,
  startedAt: 0,
};

const el = {
  audioInput: document.getElementById("audioInput"),
  refreshAudioInputs: document.getElementById("refreshAudioInputs"),
  sensitivity: document.getElementById("sensitivity"),
  sensitivityValue: document.getElementById("sensitivityValue"),
  maxEvents: document.getElementById("maxEvents"),
  typingPad: document.getElementById("typingPad"),
  startRecording: document.getElementById("startRecording"),
  stopRecording: document.getElementById("stopRecording"),
  clearResults: document.getElementById("clearResults"),
  recordingPill: document.getElementById("recordingPill"),
  predictionText: document.getElementById("predictionText"),
  status: document.getElementById("status"),
  runMeta: document.getElementById("runMeta"),
  modelMeta: document.getElementById("modelMeta"),
  eventCount: document.getElementById("eventCount"),
  events: document.getElementById("events"),
};

function setStatus(message, kind = "ok") {
  el.status.textContent = message;
  el.status.classList.toggle("error", kind === "error");
}

function expectedKeyCount() {
  const parsed = Number.parseInt(el.maxEvents.value, 10);
  if (!Number.isFinite(parsed)) return 5;
  return Math.max(1, Math.min(120, parsed));
}

function setRecording(isRecording) {
  state.recording = isRecording;
  el.startRecording.disabled = isRecording;
  el.stopRecording.disabled = !isRecording;
  el.refreshAudioInputs.disabled = isRecording;
  el.audioInput.disabled = isRecording || !state.audioDevices.length;
  el.recordingPill.classList.toggle("recording", isRecording);
  el.recordingPill.innerHTML = isRecording ? '<span class="dot"></span>Recording' : '<span class="dot"></span>Idle';
}

function audioApisAvailable() {
  return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && navigator.mediaDevices.enumerateDevices);
}

function audioDeviceLabel(device, index) {
  if (device && device.label) return device.label;
  if (device && device.deviceId === "default") return "Default microphone";
  return `Microphone ${index + 1}`;
}

async function askForAudioDevicePermission() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  for (const track of stream.getTracks()) track.stop();
}

async function refreshAudioInputs({ requestPermission = false, showStatus = false } = {}) {
  if (!audioApisAvailable()) {
    state.audioDevices = [];
    el.audioInput.innerHTML = '<option value="">Audio unavailable</option>';
    el.audioInput.disabled = true;
    return;
  }
  if (requestPermission) await askForAudioDevicePermission();

  const devices = await navigator.mediaDevices.enumerateDevices();
  state.audioDevices = devices.filter((device) => device.kind === "audioinput");
  const selectedStillAvailable = state.audioDevices.some((device) => device.deviceId === state.selectedAudioDeviceId);
  if (!selectedStillAvailable) state.selectedAudioDeviceId = state.audioDevices[0] ? state.audioDevices[0].deviceId : "";

  el.audioInput.innerHTML = "";
  if (!state.audioDevices.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No inputs found";
    el.audioInput.appendChild(option);
    el.audioInput.disabled = true;
    return;
  }

  state.audioDevices.forEach((device, index) => {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = audioDeviceLabel(device, index);
    el.audioInput.appendChild(option);
  });
  el.audioInput.value = state.selectedAudioDeviceId;
  el.audioInput.disabled = state.recording;
  if (showStatus) setStatus(`Input selected: ${el.audioInput.options[el.audioInput.selectedIndex].textContent}`);
}

function buildAudioConstraints() {
  const audioConfig = state.config.audio || {};
  const constraints = {
    channelCount: Number(audioConfig.channels || 1),
    sampleRate: Number(audioConfig.sample_rate || 48000),
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
  };
  if (state.selectedAudioDeviceId) constraints.deviceId = { exact: state.selectedAudioDeviceId };
  return constraints;
}

async function loadConfig() {
  const response = await fetch("/api/config");
  const data = await response.json();
  if (!response.ok || !data.model_exists) {
    throw new Error(data.error || "The full-dataset acoustic CNN model was not found.");
  }
  state.config = data.config || {};
  state.model = data.model_metrics || {};
  const top1 = typeof state.model.top1_accuracy === "number" ? `${(state.model.top1_accuracy * 100).toFixed(1)}%` : "n/a";
  const top5 = typeof state.model.top5_accuracy === "number" ? `${(state.model.top5_accuracy * 100).toFixed(1)}%` : "n/a";
  const segmenter = data.segmenter_exists ? "neural segmenter" : "heuristic detector";
  el.modelMeta.textContent = `${state.model.session_id || "model"} | ${segmenter} | top-1 ${top1} | top-5 ${top5}`;
  setStatus("Ready.");
}

async function startRecording() {
  if (state.recording) return;
  state.audioChunks = [];
  el.predictionText.textContent = "-";
  el.events.textContent = "";
  el.eventCount.textContent = "0 keys";
  el.runMeta.textContent = "";

  try {
    state.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: buildAudioConstraints() });
    const audioConfig = state.config.audio || {};
    state.audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: Number(audioConfig.sample_rate || 48000),
    });
    state.sampleRate = state.audioContext.sampleRate;
    state.sourceNode = state.audioContext.createMediaStreamSource(state.mediaStream);
    state.processorNode = state.audioContext.createScriptProcessor(4096, 1, 1);
    state.muteNode = state.audioContext.createGain();
    state.muteNode.gain.value = 0;
    state.processorNode.onaudioprocess = (audioEvent) => {
      if (!state.recording) return;
      state.audioChunks.push(new Float32Array(audioEvent.inputBuffer.getChannelData(0)));
    };
    state.sourceNode.connect(state.processorNode);
    state.processorNode.connect(state.muteNode);
    state.muteNode.connect(state.audioContext.destination);
  } catch (error) {
    setStatus(`Could not start recording.\n${error.message}`, "error");
    return;
  }

  state.startedAt = performance.now();
  setRecording(true);
  el.typingPad.focus();
  setStatus("Recording...");
}

async function stopRecording() {
  if (!state.recording) return;
  setRecording(false);
  await stopAudioGraph();

  if (!state.audioChunks.length) {
    setStatus("No audio captured.", "error");
    return;
  }

  const samples = mergeAudioChunks(state.audioChunks);
  const wavBytes = encodeWav(samples, state.sampleRate);
  const expectedKeys = expectedKeyCount();
  const payload = {
    audio_base64: arrayBufferToBase64(wavBytes.buffer),
    sensitivity: Number(el.sensitivity.value),
    expected_key_count: expectedKeys,
    max_events: expectedKeys,
  };

  setStatus("Decoding...");
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) {
      throw new Error(result.error || response.statusText);
    }
    renderResult(result);
  } catch (error) {
    setStatus(`Decode failed.\n${error.message}`, "error");
  }
}

async function stopAudioGraph() {
  if (state.processorNode) state.processorNode.disconnect();
  if (state.sourceNode) state.sourceNode.disconnect();
  if (state.muteNode) state.muteNode.disconnect();
  if (state.mediaStream) {
    for (const track of state.mediaStream.getTracks()) track.stop();
  }
  if (state.audioContext) await state.audioContext.close();
  state.mediaStream = null;
  state.audioContext = null;
  state.sourceNode = null;
  state.processorNode = null;
  state.muteNode = null;
}

function renderResult(result) {
  el.predictionText.textContent = result.predicted_text || "(no keys detected)";
  el.eventCount.textContent = `${result.detected_count} keys`;
  el.events.textContent = "";
  renderRunMeta(result.run);
  for (const event of result.events || []) {
    el.events.appendChild(renderEvent(event));
  }
  const limit = result.event_limit || expectedKeyCount();
  const method = result.segmentation_method === "neural_segmenter" ? "neural segmenter" : "heuristic detector";
  setStatus(`Decoded ${result.detected_count} of ${limit} expected keys with ${method} from ${result.audio_seconds.toFixed(2)} seconds of audio.`);
}

function renderRunMeta(run) {
  el.runMeta.textContent = "";
  if (!run) return;

  const saved = document.createElement("div");
  saved.className = "saved-run";

  const title = document.createElement("div");
  title.className = "saved-title";
  const label = document.createElement("strong");
  label.textContent = `Saved run ${run.run_id}`;
  const manifest = document.createElement("span");
  manifest.className = "meta";
  manifest.textContent = run.clip_manifest_path || "";
  title.append(label, manifest);
  saved.appendChild(title);

  if (run.raw_audio_url) {
    const rawLabel = document.createElement("div");
    rawLabel.className = "clip-label";
    rawLabel.textContent = "Raw recording";
    const rawAudio = document.createElement("audio");
    rawAudio.controls = true;
    rawAudio.src = run.raw_audio_url;
    saved.append(rawLabel, rawAudio);
  }

  el.runMeta.appendChild(saved);
}

function renderEvent(event) {
  const card = document.createElement("article");
  card.className = "event-card";

  const title = document.createElement("div");
  title.className = "event-title";
  const key = document.createElement("strong");
  key.textContent = event.predicted_key;
  const meta = document.createElement("span");
  meta.className = "meta";
  meta.textContent = `${event.time_seconds.toFixed(3)}s`;
  title.append(key, meta);
  card.appendChild(title);

  if (event.clip_url) {
    const clipLabel = document.createElement("div");
    clipLabel.className = "clip-label";
    clipLabel.textContent = event.clip_id || "generated clip";
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "metadata";
    audio.src = event.clip_url;
    card.append(clipLabel, audio);
  }

  for (const candidate of event.top || []) {
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = document.createElement("span");
    label.textContent = candidate.key;
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(0, Math.min(1, candidate.probability)) * 100}%`;
    track.appendChild(fill);
    const value = document.createElement("span");
    value.className = "meta";
    value.textContent = candidate.probability.toFixed(3);
    row.append(label, track, value);
    card.appendChild(row);
  }
  return card;
}

function clearResults() {
  el.predictionText.textContent = "-";
  el.events.textContent = "";
  el.eventCount.textContent = "0 keys";
  el.runMeta.textContent = "";
  setStatus("Ready.");
}

function mergeAudioChunks(chunks) {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const result = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

function encodeWav(samples, sampleRate) {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * bytesPerSample, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }
  return new Uint8Array(buffer);
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i += 1) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

el.refreshAudioInputs.addEventListener("click", () => {
  refreshAudioInputs({ requestPermission: true, showStatus: true }).catch((error) => {
    setStatus(`Could not refresh inputs.\n${error.message}`, "error");
  });
});
el.audioInput.addEventListener("change", () => {
  state.selectedAudioDeviceId = el.audioInput.value;
});
el.sensitivity.addEventListener("input", () => {
  el.sensitivityValue.textContent = Number(el.sensitivity.value).toFixed(1);
});
el.startRecording.addEventListener("click", startRecording);
el.stopRecording.addEventListener("click", stopRecording);
el.clearResults.addEventListener("click", clearResults);
el.typingPad.addEventListener("keydown", (event) => {
  if (state.recording) event.preventDefault();
});

if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener("devicechange", () => {
    refreshAudioInputs().catch(() => {});
  });
}

async function startup() {
  await loadConfig();
  await refreshAudioInputs();
}

startup().catch((error) => setStatus(`Startup failed.\n${error.message}`, "error"));
