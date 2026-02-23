import os
import asyncio
from datetime import datetime
import numpy as np

# ---------------------------------------------------------------------------
# Voice profiles: map speaker name → ChatterboxTTS generate() kwargs.
# ALEX  — slightly calmer, authoritative anchor
# JAMIE — more expressive, energetic co-host
# ---------------------------------------------------------------------------
VOICE_PROFILES = {
    "ALEX":  {"exaggeration": 0.40, "cfg_weight": 0.65, "temperature": 0.75},
    "JAMIE": {"exaggeration": 0.70, "cfg_weight": 0.35, "temperature": 0.88},
    "DEFAULT": {"exaggeration": 0.50, "cfg_weight": 0.50, "temperature": 0.80},
}

# Silence (in seconds) inserted between speaker turns
SPEAKER_PAUSE_SECS = 0.4


class TTSManager:
    """
    Manages the ChatterboxTTS model (singleton) and provides synthesis
    for both single-voice and multi-speaker dialogue use-cases.
    """
    _model = None
    _device = None

    @classmethod
    def get_model(cls):
        """Loads and returns the ChatterboxTTS model (Singleton)."""
        if cls._model is None:
            try:
                from chatterbox.tts import ChatterboxTTS
                import torch

                cls._device = "cuda" if torch.cuda.is_available() else "cpu"

                print(f"--- [TTSManager] Loading Chatterbox model on {cls._device}... ---")
                cls._model = ChatterboxTTS.from_pretrained(device=cls._device)
                print(f"--- [TTSManager] Model loaded successfully. ---")
            except ImportError as e:
                print(f"--- [TTSManager] Error: Required packages not found or structure changed: {e} ---")
                print("--- [TTSManager] Please ensure chatterbox is installed and up to date. ---")
                raise
            except Exception as e:
                print(f"--- [TTSManager] Error loading model: {e} ---")
                raise
        return cls._model

    # ------------------------------------------------------------------
    # Single-voice helper (kept for backward-compat / simple tests)
    # ------------------------------------------------------------------
    @classmethod
    async def generate_audio(cls, text: str) -> str:
        """Synthesizes speech from text with the default voice and saves to a file."""
        import torchaudio

        model = cls.get_model()

        out_dir = "output"
        os.makedirs(out_dir, exist_ok=True)
        file_name = f"{out_dir}/podcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

        print(f"--- [TTSManager] Synthesizing speech (single voice)... ---")

        try:
            profile = VOICE_PROFILES["DEFAULT"]
            wav = await asyncio.to_thread(model.generate, text, **profile)
            torchaudio.save(file_name, wav.cpu(), sample_rate=model.sr)
            print(f"--- [TTSManager] Audio saved: {file_name} ---")
            return file_name
        except Exception as e:
            print(f"--- [TTSManager] Error during synthesis: {e} ---")
            raise

    # ------------------------------------------------------------------
    # Multi-speaker dialogue synthesis
    # ------------------------------------------------------------------
    @classmethod
    async def generate_audio_dialogue(cls, segments: list) -> str:
        """
        Synthesizes a multi-speaker dialogue and concatenates into one file.

        Args:
            segments: list of (speaker, text) tuples, e.g.
                      [("ALEX", "Welcome..."), ("JAMIE", "Thanks Alex..."), ...]

        Returns:
            Path to the saved .wav file.
        """
        import torch
        import torchaudio

        model = cls.get_model()

        out_dir = "output"
        os.makedirs(out_dir, exist_ok=True)
        file_name = f"{out_dir}/podcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

        # Build a short silence tensor to pad between speaker turns
        silence_samples = int(SPEAKER_PAUSE_SECS * model.sr)
        silence = torch.zeros(1, silence_samples)

        audio_chunks = []
        for i, (speaker, text) in enumerate(segments):
            if not text.strip():
                continue

            speaker_key = speaker.upper()
            profile = VOICE_PROFILES.get(speaker_key, VOICE_PROFILES["DEFAULT"])
            print(f"--- [TTSManager] Synthesizing [{speaker_key}] segment {i+1}/{len(segments)}... ---")

            try:
                wav = await asyncio.to_thread(model.generate, text, **profile)
                wav_cpu = wav.cpu()

                # Ensure shape is (1, samples) for consistent concatenation
                if wav_cpu.dim() == 1:
                    wav_cpu = wav_cpu.unsqueeze(0)

                audio_chunks.append(wav_cpu)
                # Add silence between turns (but not after the very last one)
                if i < len(segments) - 1:
                    audio_chunks.append(silence)

            except Exception as e:
                print(f"--- [TTSManager] Error synthesizing segment {i+1} [{speaker_key}]: {e} ---")
                raise

        if not audio_chunks:
            raise ValueError("No audio segments were generated.")

        combined = torch.cat(audio_chunks, dim=1)
        torchaudio.save(file_name, combined, sample_rate=model.sr)
        print(f"--- [TTSManager] Dialogue audio saved: {file_name} ---")
        return file_name
