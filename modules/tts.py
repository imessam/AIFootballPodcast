import os
from datetime import datetime

class TTSManager:
    """
    Manages the Qwen3-TTS model to ensure it is loaded only once and
    provides an interface for speech synthesis.
    """
    _model = None
    _device = None

    @classmethod
    def get_model(cls):
        """Loads and returns the Qwen3-TTS model (Singleton)."""
        if cls._model is None:
            try:
                from qwen_tts.models import Qwen3TTS
                import torch
                
                model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
                cls._device = "cuda" if torch.cuda.is_available() else "cpu"
                
                print(f"--- [TTSManager] Loading model {model_id} on {cls._device}... ---")
                cls._model = Qwen3TTS.from_pretrained(model_id, device=cls._device)
                print(f"--- [TTSManager] Model loaded successfully. ---")
            except ImportError:
                print("--- [TTSManager] Error: Required packages (qwen-tts, torch, torchaudio) not found. ---")
                print("--- [TTSManager] Please install them using: pip install qwen-tts torch torchaudio transformers accelerate ---")
                raise
            except Exception as e:
                print(f"--- [TTSManager] Error loading model: {e} ---")
                raise
        return cls._model

    @classmethod
    async def generate_audio(cls, text: str) -> str:
        """Synthesizes speech from text and saves it to a file."""
        import torchaudio
        
        model = cls.get_model()
        
        out_dir = "output"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        file_name = f"{out_dir}/podcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        
        print(f"--- [TTSManager] Synthesizing speech... ---")
        audio_tensor = model.synthesize(text)
        
        # Save to file (Qwen3-TTS-12Hz uses 12kHz)
        torchaudio.save(file_name, audio_tensor.cpu(), sample_rate=12000)
        
        print(f"--- [TTSManager] Audio saved: {file_name} ---")
        return file_name
