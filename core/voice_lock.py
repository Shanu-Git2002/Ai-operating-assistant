"""
Voice Lock — speaker verification in front of the Live session.

Only the enrolled owner's voice reaches Gemini. Everyone else (roommates, TV,
JARVIS's own echo) is replaced by silence before it leaves the machine, so the
model never hears — and can never act on — what they said. Optionally JARVIS
then says "Sorry, you are not my owner…" (with a cooldown).

How the gate works (all local, CPU, ~25 ms per check):
  • A light energy VAD splits the mic stream into utterances.
  • The first VERIFY_SECONDS of each utterance are HELD (not sent) and checked
    against the owner's voiceprint with an ONNX speaker-embedding model
    (sherpa-onnx). Owner → the held audio is flushed and the rest streams live.
    Not the owner → the utterance is sent as silence.
  • While an accepted utterance continues, it is re-checked every few seconds,
    so a roommate who cuts in mid-sentence is muted too.
  • Very short replies ("haan", "ok") right after verified speech inherit the
    owner verdict, because they are too short to verify reliably.

The audio callback only calls process(), which does cheap numpy work and a
queue push; the model runs in this module's own worker thread.

Enrollment: say "enroll my voice" and talk naturally for ~20 seconds. The
voiceprint (an embedding, not a recording) is stored in memory/voiceprint.npy.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import numpy as np


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR        = _base_dir()
CONFIG_PATH     = BASE_DIR / "config" / "voice_lock.json"
VOICEPRINT_PATH = BASE_DIR / "memory" / "voiceprint.npy"
OTHERS_PATH     = BASE_DIR / "memory" / "voiceprints_others.npy"   # roommates (optional)
OTHER_MARGIN    = 0.05      # reject when a known other voice is this much closer
MODELS_DIR      = Path.home() / ".jarvis_models"
MODEL_NAME      = "wespeaker_en_voxceleb_resnet34_LM.onnx"
MODEL_URL       = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                   "speaker-recongition-models/" + MODEL_NAME)

SAMPLE_RATE       = 16000
VERIFY_SECONDS    = 1.2     # held at the start of each utterance for the check
RECHECK_SECONDS   = 2.5     # re-verify an accepted utterance this often
SHORT_SECONDS     = 0.45    # shorter than this cannot be verified
CONTINUITY_SECS   = 10.0    # short reply this soon after owner speech = owner
END_SILENCE_SECS  = 0.8     # silence that ends an utterance
ENROLL_SECONDS    = 20.0    # speech needed for enrollment
ENROLL_OTHER_SECS = 10.0    # speech needed to learn a roommate's voice

_DEFAULTS = {
    "enabled": False,
    "threshold": 0.55,
    "reply_on_reject": True,
    "reply_cooldown": 45,
}


# ── Config ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    try:
        return {**_DEFAULTS, **json.loads(CONFIG_PATH.read_text(encoding="utf-8"))}
    except Exception:
        return dict(_DEFAULTS)


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def model_path() -> Path:
    return MODELS_DIR / MODEL_NAME


def is_installed() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("sherpa_onnx") is not None and model_path().exists()
    except Exception:
        return False


def install(logger: Callable[[str], None] = print) -> tuple[bool, str]:
    """pip-installs sherpa-onnx and downloads the ~25 MB speaker model if missing."""
    import importlib.util
    import subprocess
    if importlib.util.find_spec("sherpa_onnx") is None:
        logger("Installing sherpa-onnx…")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "sherpa-onnx"],
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return False, "pip install sherpa-onnx failed: " + (r.stderr or r.stdout)[-200:]
    if not model_path().exists():
        logger("Downloading the speaker model (~25 MB)…")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = model_path().with_suffix(".part")
        try:
            urllib.request.urlretrieve(MODEL_URL, tmp)
            tmp.rename(model_path())
        except Exception as e:
            tmp.unlink(missing_ok=True)
            return False, f"Model download failed: {e}"
    return True, "Voice lock components are installed."


# ── Embeddings ───────────────────────────────────────────────────────────────

class _Embedder:
    def __init__(self):
        import sherpa_onnx as so
        cfg = so.SpeakerEmbeddingExtractorConfig(model=str(model_path()), num_threads=2)
        self._ex = so.SpeakerEmbeddingExtractor(cfg)

    def embed(self, pcm_f32: np.ndarray) -> np.ndarray:
        s = self._ex.create_stream()
        s.accept_waveform(SAMPLE_RATE, pcm_f32)
        s.input_finished()
        v = np.asarray(self._ex.compute(s), dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


def _to_f32(chunks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(chunks).astype(np.float32).ravel() / 32768.0


# ── The gate ─────────────────────────────────────────────────────────────────

_IDLE, _PENDING, _ACCEPTED, _REJECTED = "idle", "pending", "accepted", "rejected"


class VoiceLock:
    def __init__(self):
        self.cfg = load_config()
        self._voiceprint: Optional[np.ndarray] = None
        if VOICEPRINT_PATH.exists():
            try:
                self._voiceprint = np.load(VOICEPRINT_PATH)
            except Exception as e:
                print(f"[VoiceLock] Could not read voiceprint: {e}")

        self._others: Optional[np.ndarray] = None
        if OTHERS_PATH.exists():
            try:
                self._others = np.load(OTHERS_PATH)
            except Exception as e:
                print(f"[VoiceLock] Could not read other voiceprints: {e}")

        self._embedder: Optional[_Embedder] = None
        self._jobs: "queue.Queue[tuple]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # utterance state (touched by the audio thread; verdicts by the worker)
        self._state = _IDLE
        self._utt_id = 0
        self._held: list[np.ndarray] = []
        self._speech: list[np.ndarray] = []      # speech frames of this utterance
        self._speech_samples = 0
        self._since_check = 0
        self._silence_samples = 0
        self._submitted = False
        self._verdict: Optional[tuple[int, bool, float]] = None   # (utt_id, ok, score)
        self._recheck_verdict: Optional[tuple[int, bool, float]] = None
        self._noise = 200.0
        self._last_owner = 0.0
        self.last_score: Optional[float] = None

        # enrollment
        self._enrolling = False
        self._enroll_kind = "owner"
        self._enroll_target = ENROLL_SECONDS
        self._enroll_frames: list[np.ndarray] = []
        self._enroll_samples = 0
        self._on_enrolled: Optional[Callable[[str], None]] = None

        # hooks set by main.py
        self.on_reject: Callable[[float], None] = lambda score: None
        self.log: Callable[[str], None] = print

    # ── public state ────────────────────────────────────────────────────────
    @property
    def enrolled(self) -> bool:
        return self._voiceprint is not None

    @property
    def active(self) -> bool:
        """True when audio must go through process()."""
        return self._enrolling or (self.cfg["enabled"] and self.enrolled)

    def status(self) -> str:
        if not is_installed():
            return "Voice lock is not installed yet."
        if not self.enrolled:
            return "Voice lock is installed but no voice is enrolled yet — say 'enroll my voice'."
        state = "ON" if self.cfg["enabled"] else "OFF"
        last = f", last match score {self.last_score:.2f}" if self.last_score is not None else ""
        n_o = 0 if self._others is None else len(self._others)
        return (f"Voice lock is {state}. Threshold {self.cfg['threshold']:.2f}{last}. "
                f"Known other voices: {n_o}. "
                f"Reply to strangers: {'on' if self.cfg['reply_on_reject'] else 'off'}.")

    def set_enabled(self, on: bool) -> str:
        if on and not self.enrolled:
            return "I need to learn your voice first — say 'enroll my voice'."
        if on and not is_installed():
            ok, msg = install(self.log)
            if not ok:
                return msg
        self.cfg["enabled"] = bool(on)
        save_config(self.cfg)
        if on:
            self._ensure_worker()
        self._reset_utterance()
        return f"Voice lock {'enabled — I will only respond to your voice' if on else 'disabled'}."

    def adjust(self, delta: float) -> str:
        self.cfg["threshold"] = float(min(0.85, max(0.30, self.cfg["threshold"] + delta)))
        save_config(self.cfg)
        return f"Voice lock threshold is now {self.cfg['threshold']:.2f} ({'stricter' if delta > 0 else 'looser'})."

    def set_reply(self, on: bool) -> str:
        self.cfg["reply_on_reject"] = bool(on)
        save_config(self.cfg)
        return f"I will {'say' if on else 'not say'} 'you are not my owner' to other voices."

    def forget(self) -> str:
        self._voiceprint = None
        self._others = None
        VOICEPRINT_PATH.unlink(missing_ok=True)
        OTHERS_PATH.unlink(missing_ok=True)
        self.cfg["enabled"] = False
        save_config(self.cfg)
        return "Voiceprint deleted and voice lock disabled."

    def start_enroll(self, on_done: Callable[[str], None], kind: str = "owner") -> str:
        if kind == "other" and not self.enrolled:
            return "Enroll your own voice first, then I can learn your roommates' voices."
        if not is_installed():
            ok, msg = install(self.log)
            if not ok:
                return msg
        self._ensure_worker()
        with self._lock:
            self._enrolling = True
            self._enroll_kind = kind
            self._enroll_target = ENROLL_SECONDS if kind == "owner" else ENROLL_OTHER_SECS
            self._enroll_frames, self._enroll_samples = [], 0
            self._on_enrolled = on_done
        self._reset_utterance()
        if kind == "other":
            return (f"Ready to learn another person's voice: let ONLY your roommate talk for about "
                    f"{int(ENROLL_OTHER_SECS)} seconds now. I will ignore that voice afterwards.")
        return (f"Enrollment started: talk naturally for about {int(ENROLL_SECONDS)} seconds "
                "(tell me about your day, read something aloud). Only you should speak.")

    # ── audio thread ────────────────────────────────────────────────────────
    def process(self, frame: np.ndarray) -> list[bytes]:
        """Called from the mic callback with one int16 block. Returns the
        blocks (bytes) to send to Gemini now — possibly none, possibly a
        flush of held audio, possibly silence."""
        frame = np.asarray(frame, dtype=np.int16).reshape(-1)
        n = len(frame)
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) if n else 0.0
        is_speech = rms > max(self._noise * 2.5, 350.0)
        if not is_speech:
            self._noise = 0.95 * self._noise + 0.05 * max(rms, 50.0)

        if self._enrolling:
            if is_speech:
                self._enroll_frames.append(frame.copy())
                self._enroll_samples += n
                if self._enroll_samples >= self._enroll_target * SAMPLE_RATE:
                    self._enrolling = False
                    self._jobs.put(("enroll", list(self._enroll_frames), self._enroll_kind))
            # A roommate being learned must not reach Gemini either.
            if self._enroll_kind == "other":
                return [np.zeros(n, np.int16).tobytes()]
            return [frame.tobytes()]

        silence = np.zeros(n, dtype=np.int16).tobytes()

        # utterance boundaries
        if is_speech:
            self._silence_samples = 0
        else:
            self._silence_samples += n
        ended = self._state != _IDLE and self._silence_samples >= END_SILENCE_SECS * SAMPLE_RATE

        if self._state == _IDLE:
            if not is_speech:
                return [frame.tobytes()]
            self._start_utterance()

        if is_speech:
            self._speech.append(frame.copy())
            self._speech_samples += n

        if self._state == _PENDING:
            self._held.append(frame.copy())
            if not self._submitted and self._speech_samples >= VERIFY_SECONDS * SAMPLE_RATE:
                self._submit("verify", self._speech)
            v = self._take_verdict()
            if v is not None:
                ok, score = v
                held, self._held = self._held, []
                if ok:
                    self._state = _ACCEPTED
                    self._last_owner = time.monotonic()
                    out = [h.tobytes() for h in held]
                else:
                    self._state = _REJECTED
                    out = [np.zeros(len(h), np.int16).tobytes() for h in held]
                    if self._speech_samples >= 1.0 * SAMPLE_RATE:
                        self._fire_reject(score)
                if ended:
                    self._finish_utterance()
                return out
            if ended:
                return self._end_pending()
            return []

        if self._state == _ACCEPTED:
            if is_speech:
                self._since_check += n
                if self._since_check >= RECHECK_SECONDS * SAMPLE_RATE:
                    self._since_check = 0
                    tail = int(1.5 * SAMPLE_RATE)
                    self._submit("recheck", [_tail(self._speech, tail)])
            rv = self._take_recheck()
            if rv is not None and not rv[0]:
                self._state = _REJECTED
                self.log(f"[VoiceLock] Another voice took over (score {rv[1]:.2f}) — muting.")
                self._fire_reject(rv[1])
                return [silence]
            if ended:
                self._last_owner = time.monotonic()
                self._finish_utterance()
            return [frame.tobytes()]

        # _REJECTED
        if ended:
            self._finish_utterance()
        return [silence]

    # ── internals ───────────────────────────────────────────────────────────
    def _start_utterance(self):
        self._utt_id += 1
        self._state = _PENDING
        self._held, self._speech = [], []
        self._speech_samples = self._since_check = 0
        self._submitted = False
        with self._lock:
            self._verdict = self._recheck_verdict = None

    def _finish_utterance(self):
        self._state = _IDLE
        self._held, self._speech = [], []
        self._speech_samples = 0

    def _reset_utterance(self):
        self._finish_utterance()
        self._silence_samples = 0

    def _end_pending(self) -> list[bytes]:
        """Utterance ended before a verdict could be made."""
        if self._speech_samples >= SHORT_SECONDS * SAMPLE_RATE and not self._submitted:
            self._submit("verify", self._speech)
        if self._submitted:
            # Wait (a few frames) for the worker; keep holding meanwhile.
            v = self._wait_verdict(0.06)      # never stall the audio thread long
            if v is not None:
                ok, score = v
                held, self._held = self._held, []
                self._finish_utterance()
                if ok:
                    self._last_owner = time.monotonic()
                    return [h.tobytes() for h in held]
                return [np.zeros(len(h), np.int16).tobytes() for h in held]
            return []      # verdict will be collected on the next frame
        # Too short to verify: trust it only right after verified owner speech.
        held, self._held = self._held, []
        self._finish_utterance()
        if time.monotonic() - self._last_owner <= CONTINUITY_SECS:
            return [h.tobytes() for h in held]
        return [np.zeros(len(h), np.int16).tobytes() for h in held]

    def _submit(self, kind: str, chunks: list[np.ndarray]):
        self._submitted = self._submitted or kind == "verify"
        self._ensure_worker()
        self._jobs.put((kind, self._utt_id, list(chunks)))

    def _take_verdict(self):
        with self._lock:
            v = self._verdict
            if v and v[0] == self._utt_id:
                self._verdict = None
                return v[1], v[2]
        return None

    def _wait_verdict(self, timeout: float):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            v = self._take_verdict()
            if v is not None:
                return v
            time.sleep(0.01)
        return None

    def _take_recheck(self):
        with self._lock:
            v = self._recheck_verdict
            if v and v[0] == self._utt_id:
                self._recheck_verdict = None
                return v[1], v[2]
        return None

    def _fire_reject(self, score: float):
        try:
            self.on_reject(score)
        except Exception as e:
            print(f"[VoiceLock] on_reject failed: {e}")

    # ── worker thread ───────────────────────────────────────────────────────
    def _ensure_worker(self):
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run, daemon=True, name="VoiceLock")
        self._worker.start()

    def _run(self):
        try:
            self._embedder = _Embedder()
        except Exception as e:
            self.log(f"SYS: Voice lock could not load its model ({e}) — disabled.")
            self.cfg["enabled"] = False
            self._enrolling = False
            return
        while True:
            job = self._jobs.get()
            try:
                if job[0] == "enroll":
                    self._finish_enroll(job[1], job[2])
                    continue
                kind, utt, chunks = job
                score, other = self.scores(_to_f32(chunks))
                # a margin on re-checks so one ambiguous window does not cut the owner off
                thr = self.cfg["threshold"] - (0.08 if kind == "recheck" else 0.0)
                ok = score >= thr and other - score < OTHER_MARGIN
                self.last_score = score
                print(f"[VoiceLock] {kind} utterance {utt}: score {score:.2f} "
                      f"(closest other {other:.2f}) → "
                      f"{'owner' if ok else 'NOT owner'}")
                with self._lock:
                    if kind == "verify":
                        self._verdict = (utt, ok, score)
                    else:
                        self._recheck_verdict = (utt, ok, score)
            except Exception as e:
                print(f"[VoiceLock] worker error: {e}")

    def scores(self, pcm_f32: np.ndarray) -> tuple[float, float]:
        """(similarity to the owner, highest similarity to a known other voice)."""
        if self._voiceprint is None or self._embedder is None:
            return 0.0, -1.0
        e = self._embedder.embed(pcm_f32)
        other = float(np.max(self._others @ e)) if self._others is not None and len(self._others) else -1.0
        return float(self._voiceprint @ e), other

    def _centroid(self, pcm: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        win, hop = 3 * SAMPLE_RATE, int(1.5 * SAMPLE_RATE)
        embs = [self._embedder.embed(pcm[i:i + win])
                for i in range(0, max(1, len(pcm) - win + 1), hop)]
        c = np.mean(embs, axis=0)
        return (c / (np.linalg.norm(c) + 1e-9)).astype(np.float32), embs

    def _finish_enroll(self, frames: list[np.ndarray], kind: str = "owner"):
        pcm = _to_f32(frames)
        centroid, embs = self._centroid(pcm)
        VOICEPRINT_PATH.parent.mkdir(parents=True, exist_ok=True)

        if kind == "other":
            self._others = (centroid[None, :] if self._others is None
                            else np.vstack([self._others, centroid[None, :]]))
            np.save(OTHERS_PATH, self._others)
            msg = (f"Learned another voice ({len(self._others)} known). "
                   "I will ignore that person from now on.")
        else:
            np.save(VOICEPRINT_PATH, centroid)
            self._voiceprint = centroid
            # Calibrate: score SHORT windows (what the gate will see) against the
            # print and put the threshold just under the owner's own low end, so
            # the bar fits this voice, microphone and room.
            short, hop = int(VERIFY_SECONDS * SAMPLE_RATE), int(0.6 * SAMPLE_RATE)
            sc = [float(centroid @ self._embedder.embed(pcm[i:i + short]))
                  for i in range(0, max(1, len(pcm) - short + 1), hop)]
            thr = float(np.clip(np.percentile(sc, 10) - 0.03, 0.40, 0.75))
            self.cfg["threshold"] = round(thr, 2)
            self.cfg["enabled"] = True
            msg = (f"Voice enrolled ({len(embs)} samples, your typical match "
                   f"{np.median(sc):.2f}, threshold set to {thr:.2f}). Voice lock is now ON — "
                   "I will only respond to your voice.")
        save_config(self.cfg)
        self.log(f"SYS: {msg}")
        cb, self._on_enrolled = self._on_enrolled, None
        if cb:
            cb(msg)


def _tail(chunks: list[np.ndarray], samples: int) -> np.ndarray:
    out, total = [], 0
    for c in reversed(chunks):
        out.append(c)
        total += len(c)
        if total >= samples:
            break
    return np.concatenate(list(reversed(out)))


LOCK = VoiceLock()
