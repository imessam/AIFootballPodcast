import os
import asyncio
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Paralinguistic tags supported by ChatterboxTurboTTS (Llama tokenizer)
# These are passed through directly in text — do NOT strip them.
# ---------------------------------------------------------------------------
PARALINGUISTIC_TAGS = {
    "[laugh]", "[chuckle]", "[sigh]", "[cough]",
    "[gasp]", "[groan]", "[sniff]", "[shush]",
    "[clear throat]", "[yawn]",
}

# ---------------------------------------------------------------------------
# Voice profiles — audio_prompt_path is populated by _ensure_voice_assets()
# Turbo ignores exaggeration/cfg_weight/min_p; only temperature/top_k/top_p matter.
# ---------------------------------------------------------------------------
VOICE_PROFILES: dict = {
    "ALEX": {
        "temperature": 0.75,
        "top_k": 800,
        "top_p": 0.92,
        "repetition_penalty": 1.2,
        "audio_prompt_path": None,  # filled at runtime
    },
    "JAMIE": {
        "temperature": 0.88,
        "top_k": 1000,
        "top_p": 0.98,
        "repetition_penalty": 1.1,
        "audio_prompt_path": None,  # filled at runtime
    },
    "DEFAULT": {
        "temperature": 0.80,
        "top_k": 1000,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "audio_prompt_path": None,
    },
}

# Local asset paths for reference voice clips
_ASSETS_DIR = Path(__file__).parent.parent / "assets" / "voices"
_VOICE_FILES = {
    "ALEX":  _ASSETS_DIR / "alex.wav",
    "JAMIE": _ASSETS_DIR / "jamie.wav",
}

# HuggingFace repo that provides the reference voice samples
_HF_VOICES_REPO = "ResembleAI/chatterbox"   # fallback — we use LibriSpeech below

# Silence inserted between speaker turns (seconds)
SPEAKER_PAUSE_SECS = 0.4


class TTSManager:
    """
    Manages ChatterboxTurboTTS (singleton) with per-speaker voice cloning
    using reference WAV clips.  Supports Turbo paralinguistic tags inline.
    """
    _model = None
    _device = None

    @classmethod
    def get_model(cls):
        """Loads and returns the ChatterboxTurboTTS model (Singleton)."""
        if cls._model is None:
            try:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                import torch

                cls._device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"--- [TTSManager] Loading ChatterboxTurboTTS on {cls._device}... ---")
                cls._model = ChatterboxTurboTTS.from_pretrained(device=cls._device)
                print(f"--- [TTSManager] Turbo model loaded successfully. ---")
            except ImportError as e:
                print(f"--- [TTSManager] Import error: {e} ---")
                raise
            except Exception as e:
                print(f"--- [TTSManager] Error loading model: {e} ---")
                raise
        return cls._model

    # ------------------------------------------------------------------
    # Voice asset management
    # ------------------------------------------------------------------
    @classmethod
    def _ensure_voice_assets(cls):
        """
        Ensures alex.wav and jamie.wav exist in assets/voices/.
        If missing, downloads suitable LibriSpeech clips via HuggingFace datasets.
        Populates VOICE_PROFILES[*]['audio_prompt_path'] for each known speaker.
        """
        _ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        missing = {k: p for k, p in _VOICE_FILES.items() if not p.exists()}

        if missing:
            print(f"--- [TTSManager] Missing voice assets: {list(missing.keys())} — downloading... ---")
            cls._download_voice_samples(missing)
        else:
            print(f"--- [TTSManager] Voice assets already cached. ---")

        # Wire up audio_prompt_path in profiles
        for speaker, path in _VOICE_FILES.items():
            if path.exists() and speaker in VOICE_PROFILES:
                VOICE_PROFILES[speaker]["audio_prompt_path"] = str(path)
                print(f"--- [TTSManager] {speaker} voice → {path} ---")

    @classmethod
    def _download_voice_samples(cls, missing: dict):
        """Downloads reference clips from LibriSpeech via HuggingFace datasets."""
        try:
            import numpy as np
            import soundfile as sf
            from datasets import load_dataset

            # speaker_id → (gender, profile_key)
            target_speakers = {
                1089: "ALEX",   # male, LibriSpeech test-clean
                121:  "JAMIE",  # female, LibriSpeech test-clean
            }
            needed_profiles = {v: k for k, v in target_speakers.items() if v in missing}

            if not needed_profiles:
                return

            print("--- [TTSManager] Streaming LibriSpeech test-clean for reference clips... ---")
            ds = load_dataset(
                "openslr/librispeech_asr", "clean",
                split="test", streaming=True, trust_remote_code=True,
            )

            saved = set()
            for sample in ds:
                sid = sample["speaker_id"]
                if sid not in target_speakers:
                    continue
                profile_key = target_speakers[sid]
                if profile_key not in missing or profile_key in saved:
                    continue

                audio = sample["audio"]
                arr = np.array(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]
                duration = len(arr) / sr

                if duration >= 8:
                    out_path = _VOICE_FILES[profile_key]
                    sf.write(str(out_path), arr, sr)
                    print(f"--- [TTSManager] Saved {profile_key} voice ({duration:.1f}s) → {out_path} ---")
                    saved.add(profile_key)

                if saved == set(missing.keys()):
                    break

            if len(saved) < len(missing):
                still_missing = set(missing.keys()) - saved
                print(f"--- [TTSManager] WARNING: Could not download clips for: {still_missing}. "
                      "Place a ≥8s WAV at the path shown above to enable voice cloning. ---")

        except Exception as e:
            print(f"--- [TTSManager] Voice asset download failed: {e}. "
                  "Falling back to default Chatterbox voice. ---")

    # ------------------------------------------------------------------
    # Single-voice helper (backward compat)
    # ------------------------------------------------------------------
    @classmethod
    async def generate_audio(cls, text: str) -> str:
        """Synthesizes speech with the default (built-in) voice."""
        import torchaudio

        cls._ensure_voice_assets()
        model = cls.get_model()

        out_dir = "output"
        os.makedirs(out_dir, exist_ok=True)
        file_name = f"{out_dir}/podcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

        profile = {k: v for k, v in VOICE_PROFILES["DEFAULT"].items()
                   if k != "audio_prompt_path"}
        prompt_path = VOICE_PROFILES["DEFAULT"].get("audio_prompt_path")
        if prompt_path:
            profile["audio_prompt_path"] = prompt_path

        print(f"--- [TTSManager] Synthesising (single voice)... ---")
        try:
            wav = await asyncio.to_thread(model.generate, text, **profile)
            torchaudio.save(file_name, wav.cpu(), sample_rate=model.sr)
            print(f"--- [TTSManager] Audio saved: {file_name} ---")
            return file_name
        except Exception as e:
            print(f"--- [TTSManager] Error: {e} ---")
            raise

    # ------------------------------------------------------------------
    # Multi-speaker dialogue synthesis
    # ------------------------------------------------------------------
    @classmethod
    async def generate_audio_dialogue(cls, segments: list) -> str:
        """
        Synthesises a multi-speaker dialogue and concatenates into one WAV.

        Args:
            segments: list of (speaker, text) tuples.
                      Speaker names must match keys in VOICE_PROFILES.

        Returns:
            Path to the saved .wav file.
        """
        import torch
        import torchaudio

        cls._ensure_voice_assets()
        model = cls.get_model()

        out_dir = "output"
        os.makedirs(out_dir, exist_ok=True)
        file_name = f"{out_dir}/podcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

        silence_samples = int(SPEAKER_PAUSE_SECS * model.sr)
        silence = torch.zeros(1, silence_samples)

        audio_chunks = []

        for i, (speaker, text) in enumerate(segments):
            if not text.strip():
                continue

            speaker_key = speaker.upper()
            raw_profile = VOICE_PROFILES.get(speaker_key, VOICE_PROFILES["DEFAULT"])

            # Build kwargs — exclude None audio_prompt_path
            profile_kwargs = {k: v for k, v in raw_profile.items()
                              if k != "audio_prompt_path" and v is not None}
            prompt = raw_profile.get("audio_prompt_path")
            if prompt:
                profile_kwargs["audio_prompt_path"] = prompt

            print(f"--- [TTSManager] [{speaker_key}] seg {i+1}/{len(segments)}"
                  f" | voice={'custom' if prompt else 'default'}"
                  f" | text: {text[:60]}... ---")

            try:
                wav = await asyncio.to_thread(model.generate, text, **profile_kwargs)
                wav_cpu = wav.cpu()
                if wav_cpu.dim() == 1:
                    wav_cpu = wav_cpu.unsqueeze(0)
                audio_chunks.append(wav_cpu)
                if i < len(segments) - 1:
                    audio_chunks.append(silence)
            except Exception as e:
                print(f"--- [TTSManager] Error on segment {i+1}: {e} ---")
                raise

        if not audio_chunks:
            raise ValueError("No audio segments were generated.")

        combined = torch.cat(audio_chunks, dim=1)
        torchaudio.save(file_name, combined, sample_rate=model.sr)
        print(f"--- [TTSManager] Dialogue saved: {file_name} ---")
        return file_name
