import os
from datetime import datetime
import numpy as np

class TTSManager:
    """
    Manages the ChatterboxTTS model to ensure it is loaded only once and
    provides an interface for speech synthesis.
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
        
        try:
            wav = model.generate(text)
            
            # Save to file
            torchaudio.save(file_name, wav, sample_rate=model.sr)
            
            print(f"--- [TTSManager] Audio saved: {file_name} ---")
            return file_name
            
        except Exception as e:
            print(f"--- [TTSManager] Error during synthesis: {e} ---")
            raise
