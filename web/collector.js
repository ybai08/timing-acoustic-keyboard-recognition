const state = {
  config: {},
  promptSets: {},
  promptSet: "",
  promptIndex: 0,
  recording: false,
  audioContext: null,
  mediaStream: null,
  sourceNode: null,
  processorNode: null,
  muteNode: null,
  audioChunks: [],
  audioDevices: [],
  selectedAudioDeviceId: "",
  requestedAudioDevice: null,
  activeAudioDevice: null,
  activeAudioTrackSettings: {},
  savingTrial: false,
  deletingTrial: false,
  lastSavedTrial: null,
  sampleRate: 48000,
  trialStartPerf: 0,
  startedAt: "",
  events: [],
  nextEventIndex: 0,
  keyStacks: new Map(),
};

const el = {
  participantId: document.getElementById("participantId"),
  sessionId: document.getElementById("sessionId"),
  promptSet: document.getElementById("promptSet"),
  promptSetSummary: document.getElementById("promptSetSummary"),
  audioInput: document.getElementById("audioInput"),
  audioInputSummary: document.getElementById("audioInputSummary"),
  audioInputHint: document.getElementById("audioInputHint"),
  refreshAudioInputs: document.getElementById("refreshAudioInputs"),
  promptIndex: document.getElementById("promptIndex"),
  promptText: document.getElementById("promptText"),
  typingBox: document.getElementById("typingBox"),
  startTrial: document.getElementById("startTrial"),
  stopTrial: document.getElementById("stopTrial"),
  clearText: document.getElementById("clearText"),
  deleteLastTrial: document.getElementById("deleteLastTrial"),
  prevPrompt: document.getElementById("prevPrompt"),
  nextPrompt: document.getElementById("nextPrompt"),
  status: document.getElementById("status"),
  recordingIndicator: document.getElementById("recordingIndicator"),
};

function setStatus(message) {
  el.status.textContent = message;
}

function setRecordingIndicator(label, isRecording) {
  const text = el.recordingIndicator.querySelector("span:last-child");
  text.textContent = label;
  el.recordingIndicator.classList.toggle("recording", isRecording);
  el.status.classList.toggle("recording-status", isRecording);
}

function updateDeleteLastTrialButton() {
  const trial = state.lastSavedTrial;
  const canDelete = Boolean(trial && !state.recording && !state.savingTrial && !state.deletingTrial);
  el.deleteLastTrial.disabled = !canDelete;
  el.deleteLastTrial.textContent = trial ? `Delete ${trial.trial_id}` : "Delete Last Trial";
}

function currentPrompts() {
  return state.promptSets[state.promptSet] || [""];
}

function currentPrompt() {
  const prompts = currentPrompts();
  return prompts[state.promptIndex % prompts.length];
}

function refreshPrompt() {
  const prompts = currentPrompts();
  state.promptIndex = ((state.promptIndex % prompts.length) + prompts.length) % prompts.length;
  el.promptIndex.value = String(state.promptIndex + 1);
  el.promptText.textContent = `${state.promptIndex + 1}/${prompts.length}: ${prompts[state.promptIndex]}`;
  el.promptSetSummary.textContent = state.promptSet.replaceAll("_", " ");
}

function sanitizeId(value) {
  const cleaned = value.trim().replace(/[^A-Za-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "");
  return cleaned || "unknown";
}

function audioApisAvailable() {
  return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && navigator.mediaDevices.enumerateDevices);
}

function audioDeviceLabel(device, index) {
  if (device && device.label) return device.label;
  if (device && device.deviceId === "default") return "Default microphone";
  return `Microphone ${index + 1}`;
}

function audioDeviceInfo(deviceId) {
  const index = state.audioDevices.findIndex((device) => device.deviceId === deviceId);
  const device = index >= 0 ? state.audioDevices[index] : null;
  if (!device) {
    return {
      device_id: deviceId || "",
      group_id: "",
      label: deviceId ? "Selected microphone" : "Default browser microphone",
    };
  }
  return {
    device_id: device.deviceId,
    group_id: device.groupId || "",
    label: audioDeviceLabel(device, index),
  };
}

function updateAudioInputSummary() {
  if (!state.audioDevices.length) {
    el.audioInputSummary.textContent = "No microphone detected";
    return;
  }
  el.audioInputSummary.textContent = audioDeviceInfo(state.selectedAudioDeviceId).label;
}

function setAudioInputControlsEnabled(enabled) {
  el.audioInput.disabled = !enabled || !state.audioDevices.length;
  el.refreshAudioInputs.disabled = !enabled;
}

function renderAudioDevices() {
  el.audioInput.innerHTML = "";

  if (!state.audioDevices.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No audio inputs found";
    el.audioInput.appendChild(option);
    el.audioInputHint.textContent = "Connect a microphone, then click Allow / Refresh.";
    setAudioInputControlsEnabled(!state.recording);
    updateAudioInputSummary();
    return;
  }

  state.audioDevices.forEach((device, index) => {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = audioDeviceLabel(device, index);
    el.audioInput.appendChild(option);
  });
  el.audioInput.value = state.selectedAudioDeviceId;

  const labelsVisible = state.audioDevices.some((device) => device.label);
  el.audioInputHint.textContent = labelsVisible
    ? "Recording will use the selected input."
    : "Device names may appear after microphone access is allowed.";
  setAudioInputControlsEnabled(!state.recording);
  updateAudioInputSummary();
}

async function askForAudioDevicePermission() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  for (const track of stream.getTracks()) track.stop();
}

async function refreshAudioInputs({ requestPermission = false, showStatus = false, preferredDeviceId = "" } = {}) {
  if (!audioApisAvailable()) {
    state.audioDevices = [];
    el.audioInput.innerHTML = '<option value="">Audio device selection unsupported</option>';
    el.audioInputHint.textContent = "This browser cannot list microphone inputs.";
    setAudioInputControlsEnabled(false);
    updateAudioInputSummary();
    return;
  }
  if (state.recording) return;

  let permissionError = null;
  if (requestPermission) {
    try {
      await askForAudioDevicePermission();
    } catch (error) {
      permissionError = error;
    }
  }

  const devices = await navigator.mediaDevices.enumerateDevices();
  state.audioDevices = devices.filter((device) => device.kind === "audioinput");
  const currentDeviceId = preferredDeviceId || state.selectedAudioDeviceId || el.audioInput.value;
  const currentStillAvailable = state.audioDevices.some((device) => device.deviceId === currentDeviceId);
  state.selectedAudioDeviceId = currentStillAvailable
    ? currentDeviceId
    : (state.audioDevices[0] ? state.audioDevices[0].deviceId : "");
  renderAudioDevices();

  if (showStatus && permissionError) {
    setStatus(`Could not access microphone devices.\n${permissionError.message}`);
  } else if (showStatus && state.audioDevices.length) {
    setStatus(`Audio inputs refreshed. Selected: ${audioDeviceInfo(state.selectedAudioDeviceId).label}`);
  }
}

function buildAudioConstraints(audioConfig) {
  const constraints = {
    channelCount: Number(audioConfig.channels || 1),
    sampleRate: Number(audioConfig.sample_rate || 48000),
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
  };
  if (state.selectedAudioDeviceId) {
    constraints.deviceId = { exact: state.selectedAudioDeviceId };
  }
  return constraints;
}

function summarizeTrackSettings(settings) {
  return {
    device_id: settings.deviceId || "",
    group_id: settings.groupId || "",
    sample_rate: settings.sampleRate || null,
    channel_count: settings.channelCount || null,
    echo_cancellation: settings.echoCancellation ?? null,
    noise_suppression: settings.noiseSuppression ?? null,
    auto_gain_control: settings.autoGainControl ?? null,
  };
}

async function loadConfig() {
  const response = await fetch("/api/config");
  if (!response.ok) throw new Error("Could not load config from local server.");
  const data = await response.json();
  state.config = data.config || {};
  state.promptSets = data.prompt_sets || {};
  state.promptSet = Object.keys(state.promptSets)[0] || "default";
  el.sessionId.value = data.default_session_id || "session_001";
  el.promptSet.innerHTML = "";
  for (const name of Object.keys(state.promptSets)) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    el.promptSet.appendChild(option);
  }
  el.promptSet.value = state.promptSet;
  refreshPrompt();
  setStatus("Ready. Click Start Trial, type in the box, then click Stop + Save.");
}

function recordKeyEvent(event, eventType) {
  if (!state.recording) return;
  const now = performance.now();
  let eventIndex;
  if (eventType === "keydown") {
    eventIndex = state.nextEventIndex++;
    const stack = state.keyStacks.get(event.code) || [];
    stack.push(eventIndex);
    state.keyStacks.set(event.code, stack);
  } else {
    const stack = state.keyStacks.get(event.code) || [];
    eventIndex = stack.length ? stack.pop() : state.nextEventIndex;
  }
  state.events.push({
    event_index: eventIndex,
    event_type: eventType,
    key: event.key && event.key.length === 1 ? event.key.toLowerCase() : event.key,
    char: event.key && event.key.length === 1 ? event.key : "",
    keysym: event.key,
    code: event.code,
    keycode: event.keyCode,
    location: event.location,
    repeat: event.repeat,
    browser_time_ms: now.toFixed(3),
    trial_elapsed_seconds: ((now - state.trialStartPerf) / 1000).toFixed(9),
  });
}

async function startTrial() {
  if (state.recording) return;
  el.participantId.value = sanitizeId(el.participantId.value);
  el.sessionId.value = sanitizeId(el.sessionId.value);
  el.typingBox.value = "";
  state.events = [];
  state.audioChunks = [];
  state.keyStacks = new Map();
  state.nextEventIndex = 0;
  state.requestedAudioDevice = null;
  state.activeAudioDevice = null;
  state.activeAudioTrackSettings = {};

  try {
    const audioConfig = state.config.audio || {};
    const requestedSampleRate = Number(audioConfig.sample_rate || 48000);
    state.requestedAudioDevice = audioDeviceInfo(state.selectedAudioDeviceId);
    state.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: buildAudioConstraints(audioConfig),
    });
    const audioTrack = state.mediaStream.getAudioTracks()[0];
    const trackSettings = audioTrack && audioTrack.getSettings ? audioTrack.getSettings() : {};
    state.activeAudioTrackSettings = summarizeTrackSettings(trackSettings);
    if (trackSettings.deviceId) {
      state.selectedAudioDeviceId = trackSettings.deviceId;
      await refreshAudioInputs({ preferredDeviceId: trackSettings.deviceId }).catch(() => {});
    }
    state.activeAudioDevice = audioDeviceInfo(trackSettings.deviceId || state.selectedAudioDeviceId);
    state.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: requestedSampleRate });
    state.sampleRate = state.audioContext.sampleRate;
    state.sourceNode = state.audioContext.createMediaStreamSource(state.mediaStream);
    state.processorNode = state.audioContext.createScriptProcessor(4096, 1, 1);
    state.muteNode = state.audioContext.createGain();
    state.muteNode.gain.value = 0;
    state.processorNode.onaudioprocess = (audioEvent) => {
      if (!state.recording) return;
      const input = audioEvent.inputBuffer.getChannelData(0);
      state.audioChunks.push(new Float32Array(input));
    };
    state.sourceNode.connect(state.processorNode);
    state.processorNode.connect(state.muteNode);
    state.muteNode.connect(state.audioContext.destination);
  } catch (error) {
    setStatus(`Could not start microphone recording.\n${error.message}\n\nCheck browser microphone permission and your input device.`);
    return;
  }

  state.recording = true;
  state.trialStartPerf = performance.now();
  state.startedAt = new Date().toISOString();
  el.startTrial.disabled = true;
  el.stopTrial.disabled = false;
  el.prevPrompt.disabled = true;
  el.nextPrompt.disabled = true;
  setAudioInputControlsEnabled(false);
  setRecordingIndicator("Recording", true);
  updateDeleteLastTrialButton();
  el.typingBox.focus();
  setStatus(`Recording from ${state.activeAudioDevice.label}. Type the prompt in the text box, then click Stop + Save.`);
}

async function stopAndSave() {
  if (!state.recording) return;
  const endedAt = new Date().toISOString();
  const durationSeconds = (performance.now() - state.trialStartPerf) / 1000;
  const savedPromptIndex = state.promptIndex;
  state.recording = false;
  state.savingTrial = true;

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

  el.startTrial.disabled = true;
  el.stopTrial.disabled = true;
  el.prevPrompt.disabled = true;
  el.nextPrompt.disabled = true;
  setAudioInputControlsEnabled(true);
  setRecordingIndicator("Idle", false);
  updateDeleteLastTrialButton();

  if (!state.audioChunks.length) {
    state.savingTrial = false;
    el.startTrial.disabled = false;
    el.prevPrompt.disabled = false;
    el.nextPrompt.disabled = false;
    updateDeleteLastTrialButton();
    setStatus("No audio was captured. Try again and check microphone permission.");
    return;
  }

  const samples = mergeAudioChunks(state.audioChunks);
  const wavBytes = encodeWav(samples, state.sampleRate);
  const audioBase64 = arrayBufferToBase64(wavBytes.buffer);
  const payload = {
    participant_id: el.participantId.value,
    session_id: el.sessionId.value,
    prompt_set: state.promptSet,
    prompt_index: state.promptIndex,
    prompt_text: currentPrompt(),
    typed_text: el.typingBox.value,
    started_at: state.startedAt,
    ended_at: endedAt,
    duration_seconds: Number(durationSeconds.toFixed(6)),
    sample_rate: state.sampleRate,
    channels: 1,
    audio_frame_count: samples.length,
    audio_input_device: {
      requested: state.requestedAudioDevice,
      active: state.activeAudioDevice,
      track_settings: state.activeAudioTrackSettings,
    },
    audio_base64: audioBase64,
    events: state.events,
  };

  setStatus("Saving trial...");
  let response;
  let result;
  try {
    response = await fetch("/api/save-trial", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    result = await response.json();
  } catch (error) {
    state.savingTrial = false;
    el.startTrial.disabled = false;
    el.prevPrompt.disabled = false;
    el.nextPrompt.disabled = false;
    updateDeleteLastTrialButton();
    setStatus(`Save failed: ${error.message}`);
    return;
  }
  if (!response.ok || !result.ok) {
    state.savingTrial = false;
    el.startTrial.disabled = false;
    el.prevPrompt.disabled = false;
    el.nextPrompt.disabled = false;
    updateDeleteLastTrialButton();
    setStatus(`Save failed: ${result.error || response.statusText}`);
    return;
  }
  state.lastSavedTrial = {
    trial_id: result.trial_id,
    session_id: result.session_id,
    session_dir: result.session_dir,
    prompt_set: payload.prompt_set,
    prompt_index: savedPromptIndex,
  };
  state.savingTrial = false;
  el.startTrial.disabled = false;
  el.typingBox.value = "";
  state.promptIndex += 1;
  refreshPrompt();
  el.prevPrompt.disabled = false;
  el.nextPrompt.disabled = false;
  updateDeleteLastTrialButton();
  setStatus(`Saved ${result.trial_id}\n${result.session_dir}\nEvents: ${state.events.length}\nAudio frames: ${samples.length}`);
}

async function deleteLastSavedTrial() {
  const trial = state.lastSavedTrial;
  if (!trial || state.recording || state.savingTrial || state.deletingTrial) return;

  const confirmed = window.confirm(
    `Delete ${trial.trial_id}?\n\nThis removes the raw WAV, events CSV, and metadata JSON files for that trial.`
  );
  if (!confirmed) return;

  state.deletingTrial = true;
  updateDeleteLastTrialButton();
  setStatus(`Deleting ${trial.trial_id}...`);

  let response;
  let result;
  try {
    response = await fetch("/api/delete-trial", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: trial.session_id,
        trial_id: trial.trial_id,
      }),
    });
    result = await response.json();
  } catch (error) {
    state.deletingTrial = false;
    updateDeleteLastTrialButton();
    setStatus(`Delete failed: ${error.message}`);
    return;
  }

  if (!response.ok || !result.ok) {
    state.deletingTrial = false;
    updateDeleteLastTrialButton();
    setStatus(`Delete failed: ${result.error || response.statusText}`);
    return;
  }

  state.lastSavedTrial = null;
  state.deletingTrial = false;
  el.typingBox.value = "";
  if (Number.isInteger(trial.prompt_index)) {
    if (trial.prompt_set && state.promptSets[trial.prompt_set]) {
      state.promptSet = trial.prompt_set;
      el.promptSet.value = trial.prompt_set;
    }
    state.promptIndex = trial.prompt_index;
    refreshPrompt();
  }
  updateDeleteLastTrialButton();
  setStatus(`Deleted ${result.trial_id}\nRemoved files: ${result.deleted_paths.length}\nPrompt reset so you can retry.`);
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

el.promptSet.addEventListener("change", () => {
  state.promptSet = el.promptSet.value;
  state.promptIndex = 0;
  refreshPrompt();
});
el.prevPrompt.addEventListener("click", () => {
  state.promptIndex -= 1;
  refreshPrompt();
});
el.nextPrompt.addEventListener("click", () => {
  state.promptIndex += 1;
  refreshPrompt();
});
el.clearText.addEventListener("click", () => {
  if (!state.recording) el.typingBox.value = "";
});
el.deleteLastTrial.addEventListener("click", deleteLastSavedTrial);
el.audioInput.addEventListener("change", () => {
  state.selectedAudioDeviceId = el.audioInput.value;
  updateAudioInputSummary();
});
el.refreshAudioInputs.addEventListener("click", () => {
  refreshAudioInputs({ requestPermission: true, showStatus: true })
    .catch((error) => setStatus(`Could not refresh audio inputs.\n${error.message}`));
});
el.startTrial.addEventListener("click", startTrial);
el.stopTrial.addEventListener("click", stopAndSave);
el.typingBox.addEventListener("keydown", (event) => recordKeyEvent(event, "keydown"));
el.typingBox.addEventListener("keyup", (event) => recordKeyEvent(event, "keyup"));

if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener("devicechange", () => {
    refreshAudioInputs().catch(() => {});
  });
}

async function startup() {
  await loadConfig();
  await refreshAudioInputs();
  updateDeleteLastTrialButton();
}

startup().catch((error) => setStatus(`Startup failed: ${error.message}`));
