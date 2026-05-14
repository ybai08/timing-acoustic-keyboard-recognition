from __future__ import annotations

import queue
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf

from keyboard_fusion.collection import (
    build_trial_paths,
    load_prompt_files,
    make_session_id,
    next_trial_id,
    sanitize_id,
    write_events_csv,
    write_metadata_json,
)
from keyboard_fusion.config import load_config
from keyboard_fusion.paths import RAW_DATA_DIR


class CollectionApp:
    """Small foreground app for collecting audio plus key event logs."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Keyboard Fusion Data Collector")
        self.root.geometry("900x620")

        self.config = load_config()
        self.prompt_sets = load_prompt_files()
        if not self.prompt_sets:
            self.prompt_sets = {"default": ["the quick brown fox"]}

        self.participant_var = tk.StringVar(value="p001")
        self.session_var = tk.StringVar(value=make_session_id())
        self.prompt_set_var = tk.StringVar(value=next(iter(self.prompt_sets)))
        self.prompt_index_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="Ready. Click Start Trial, then type inside the box.")

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.audio_blocks: list[np.ndarray] = []
        self.audio_statuses: list[str] = []
        self.stream: sd.InputStream | None = None

        self.events: list[dict[str, Any]] = []
        self.keydown_stacks: dict[str, list[int]] = {}
        self.next_event_index = 0
        self.recording_active = False
        self.trial_start_monotonic = 0.0
        self.trial_started_at_iso = ""

        self._build_ui()
        self._refresh_prompt()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill=tk.BOTH, expand=True)

        setup = ttk.LabelFrame(container, text="Trial Setup", padding=12)
        setup.pack(fill=tk.X)

        ttk.Label(setup, text="Participant ID").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(setup, textvariable=self.participant_var, width=18).grid(row=1, column=0, sticky=tk.W, padx=(0, 12))

        ttk.Label(setup, text="Session ID").grid(row=0, column=1, sticky=tk.W)
        ttk.Entry(setup, textvariable=self.session_var, width=28).grid(row=1, column=1, sticky=tk.W, padx=(0, 12))

        ttk.Label(setup, text="Prompt Set").grid(row=0, column=2, sticky=tk.W)
        prompt_combo = ttk.Combobox(
            setup,
            textvariable=self.prompt_set_var,
            values=list(self.prompt_sets),
            state="readonly",
            width=24,
        )
        prompt_combo.grid(row=1, column=2, sticky=tk.W, padx=(0, 12))
        prompt_combo.bind("<<ComboboxSelected>>", self._on_prompt_set_changed)

        controls = ttk.Frame(setup)
        controls.grid(row=1, column=3, sticky=tk.E)
        ttk.Button(controls, text="Previous Prompt", command=self.previous_prompt).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controls, text="Next Prompt", command=self.next_prompt).pack(side=tk.LEFT)

        setup.columnconfigure(4, weight=1)

        prompt_frame = ttk.LabelFrame(container, text="Prompt To Type", padding=12)
        prompt_frame.pack(fill=tk.X, pady=(14, 0))
        self.prompt_label = ttk.Label(prompt_frame, text="", wraplength=820, font=("Helvetica", 18, "bold"))
        self.prompt_label.pack(anchor=tk.W)

        typing_frame = ttk.LabelFrame(container, text="Type Here During Recording", padding=12)
        typing_frame.pack(fill=tk.BOTH, expand=True, pady=(14, 0))
        self.text = tk.Text(typing_frame, height=8, wrap=tk.WORD, font=("Menlo", 15))
        self.text.pack(fill=tk.BOTH, expand=True)
        self.text.bind("<KeyPress>", self._on_key_press)
        self.text.bind("<KeyRelease>", self._on_key_release)

        button_row = ttk.Frame(container)
        button_row.pack(fill=tk.X, pady=(14, 0))
        self.start_button = ttk.Button(button_row, text="Start Trial", command=self.start_trial)
        self.start_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(button_row, text="Stop + Save", command=self.stop_and_save, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(button_row, text="Clear Text", command=self.clear_text).pack(side=tk.LEFT, padx=(8, 0))

        status = ttk.Label(container, textvariable=self.status_var, wraplength=850)
        status.pack(fill=tk.X, pady=(12, 0))

        note = ttk.Label(
            container,
            text=(
                "This app records only after Start Trial is clicked and logs only keys typed in this text box. "
                "Use synthetic prompts only."
            ),
            foreground="#555555",
            wraplength=850,
        )
        note.pack(fill=tk.X, pady=(6, 0))

    def _current_prompt_list(self) -> list[str]:
        return self.prompt_sets[self.prompt_set_var.get()]

    def _current_prompt(self) -> str:
        prompts = self._current_prompt_list()
        index = self.prompt_index_var.get() % len(prompts)
        return prompts[index]

    def _refresh_prompt(self) -> None:
        prompts = self._current_prompt_list()
        index = self.prompt_index_var.get() % len(prompts)
        self.prompt_index_var.set(index)
        self.prompt_label.configure(text=f"{index + 1}/{len(prompts)}: {prompts[index]}")

    def _on_prompt_set_changed(self, _event: tk.Event[Any] | None = None) -> None:
        self.prompt_index_var.set(0)
        self._refresh_prompt()

    def previous_prompt(self) -> None:
        if self.recording_active:
            return
        self.prompt_index_var.set(self.prompt_index_var.get() - 1)
        self._refresh_prompt()

    def next_prompt(self) -> None:
        if self.recording_active:
            return
        self.prompt_index_var.set(self.prompt_index_var.get() + 1)
        self._refresh_prompt()

    def clear_text(self) -> None:
        if not self.recording_active:
            self.text.delete("1.0", tk.END)

    def _audio_callback(self, indata: np.ndarray, _frames: int, _time_info: Any, status: sd.CallbackFlags) -> None:
        if status:
            self.audio_statuses.append(str(status))
        self.audio_queue.put(indata.copy())

    def start_trial(self) -> None:
        if self.recording_active:
            return

        participant_id = sanitize_id(self.participant_var.get())
        session_id = sanitize_id(self.session_var.get())
        if not participant_id or not session_id:
            messagebox.showerror("Missing setup", "Participant ID and Session ID are required.")
            return

        self.participant_var.set(participant_id)
        self.session_var.set(session_id)
        self.text.delete("1.0", tk.END)
        self.events = []
        self.keydown_stacks = {}
        self.next_event_index = 0
        self.audio_blocks = []
        self.audio_statuses = []
        while not self.audio_queue.empty():
            self.audio_queue.get()

        audio_config = self.config.get("audio", {})
        sample_rate = int(audio_config.get("sample_rate", 48000))
        channels = int(audio_config.get("channels", 1))
        dtype = str(audio_config.get("dtype", "float32"))

        try:
            self.stream = sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype=dtype,
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception as exc:
            self.stream = None
            messagebox.showerror(
                "Could not start audio recording",
                f"{exc}\n\nCheck that your microphone is connected and that Terminal/VS Code has microphone permission.",
            )
            return

        self.recording_active = True
        self.trial_start_monotonic = time.perf_counter()
        self.trial_started_at_iso = datetime.now().isoformat(timespec="seconds")
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.text.focus_set()
        self.status_var.set("Recording. Type the prompt in the text box, then click Stop + Save.")

    def _record_event(self, event: tk.Event[Any], event_type: str) -> None:
        if not self.recording_active:
            return

        now = time.perf_counter()
        keysym = str(event.keysym)
        char = event.char if event.char else ""
        key = char.lower() if char and len(char) == 1 else keysym

        if event_type == "keydown":
            event_index = self.next_event_index
            self.next_event_index += 1
            self.keydown_stacks.setdefault(keysym, []).append(event_index)
        else:
            stack = self.keydown_stacks.get(keysym, [])
            event_index = stack.pop() if stack else self.next_event_index

        self.events.append(
            {
                "event_index": event_index,
                "event_type": event_type,
                "key": key,
                "char": char,
                "keysym": keysym,
                "keycode": event.keycode,
                "timestamp_monotonic": f"{now:.9f}",
                "trial_elapsed_seconds": f"{now - self.trial_start_monotonic:.9f}",
            }
        )

    def _on_key_press(self, event: tk.Event[Any]) -> None:
        self._record_event(event, "keydown")

    def _on_key_release(self, event: tk.Event[Any]) -> None:
        self._record_event(event, "keyup")

    def stop_and_save(self) -> None:
        if not self.recording_active:
            return

        ended_at = datetime.now().isoformat(timespec="seconds")
        duration_seconds = time.perf_counter() - self.trial_start_monotonic

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        while not self.audio_queue.empty():
            self.audio_blocks.append(self.audio_queue.get())

        self.recording_active = False
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)

        if not self.audio_blocks:
            messagebox.showerror("No audio captured", "The microphone stream did not return audio blocks.")
            self.status_var.set("No audio captured. Check the selected microphone and try again.")
            return

        audio = np.concatenate(self.audio_blocks, axis=0)
        audio_config = self.config.get("audio", {})
        sample_rate = int(audio_config.get("sample_rate", 48000))

        session_id = sanitize_id(self.session_var.get())
        session_dir = RAW_DATA_DIR / "sessions" / session_id
        trial_id = next_trial_id(session_dir)
        paths = build_trial_paths(session_id, trial_id)
        paths.session_dir.mkdir(parents=True, exist_ok=True)

        sf.write(paths.audio_path, audio, sample_rate)
        write_events_csv(paths.events_path, self.events)

        typed_text = self.text.get("1.0", tk.END).rstrip("\n")
        metadata = {
            "trial_id": trial_id,
            "session_id": session_id,
            "participant_id": sanitize_id(self.participant_var.get()),
            "prompt_set": self.prompt_set_var.get(),
            "prompt_index": self.prompt_index_var.get(),
            "prompt_text": self._current_prompt(),
            "typed_text": typed_text,
            "started_at": self.trial_started_at_iso,
            "ended_at": ended_at,
            "duration_seconds": round(duration_seconds, 6),
            "audio_file_path": str(paths.audio_path.relative_to(paths.session_dir)),
            "events_file_path": str(paths.events_path.relative_to(paths.session_dir)),
            "sample_rate": sample_rate,
            "channels": int(audio_config.get("channels", 1)),
            "dtype": str(audio_config.get("dtype", "float32")),
            "event_count": len(self.events),
            "audio_frame_count": int(audio.shape[0]),
            "keyboard": self.config.get("hardware", {}).get("keyboard", {}),
            "microphone": self.config.get("hardware", {}).get("microphone", {}),
            "environment": self.config.get("environment", {}),
            "audio_statuses": self.audio_statuses,
            "notes": (
                "Key event timestamps are seconds elapsed from trial_start_monotonic. "
                "Audio sample 0 is aligned to the same Start Trial action."
            ),
        }
        write_metadata_json(paths.metadata_path, metadata)

        self.status_var.set(
            f"Saved {trial_id} in {paths.session_dir}. Audio frames: {audio.shape[0]}, events: {len(self.events)}."
        )
        messagebox.showinfo(
            "Trial saved",
            f"Saved:\n{paths.audio_path.name}\n{paths.events_path.name}\n{paths.metadata_path.name}",
        )
        self.next_prompt()


def main() -> int:
    root = tk.Tk()
    CollectionApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

