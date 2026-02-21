import os
from datetime import datetime
import numpy as np

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
                from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
                import torch
                
                model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
                cls._device = "cuda" if torch.cuda.is_available() else "cpu"
                
                print(f"--- [TTSManager] Loading model {model_id} on {cls._device}... ---")
                # Qwen3-TTS models are large, using bfloat16 or float16 is recommended
                cls._model = Qwen3TTSModel.from_pretrained(
                    model_id, 
                    device_map=cls._device,
                    torch_dtype=torch.bfloat16 if cls._device == "cuda" else torch.float32
                )
                print(f"--- [TTSManager] Model loaded successfully. ---")
            except ImportError as e:
                print(f"--- [TTSManager] Error: Required packages not found or structure changed: {e} ---")
                print("--- [TTSManager] Please ensure qwen-tts is installed and up to date. ---")
                raise
            except Exception as e:
                print(f"--- [TTSManager] Error loading model: {e} ---")
                raise
        return cls._model

    @classmethod
    async def generate_audio(cls, text: str) -> str:
        """Synthesizes speech from text and saves it to a file."""
        import torch
        import torchaudio
        
        model_wrapper = cls.get_model()
        
        out_dir = "output"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        file_name = f"{out_dir}/podcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        
        print(f"--- [TTSManager] Synthesizing speech... ---")
        
        # For the 12Hz-Base model, we use generate_voice_clone or similar.
        # Since it's a base model, it might need a reference audio for cloning,
        # but usually it has a default if not provided, or we use a simple generate call.
        # Let's try to use the most appropriate method based on the model type.
        
        try:
            if model_wrapper.model.tts_model_type == "base":
                # For base model, we might need a default or use a simple generate if supported
                # Here we assume it can generate without ref if it has a default
                wavs, sr = model_wrapper.generate_voice_clone(text=text)
            elif model_wrapper.model.tts_model_type == "voice_design":
                wavs, sr = model_wrapper.generate_voice_design(text=text, instruct="")
            else:
                # Fallback to a generic generate if available on the underlying model
                # or try custom voice with a placeholder
                wavs, sr = model_wrapper.generate_custom_voice(text=text, speaker="female_1")
            
            # wavs is a list of np.ndarray, we take the first one
            audio_np = wavs[0]
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0) # (1, T)
            
            # Save to file
            torchaudio.save(file_name, audio_tensor, sample_rate=sr)
            
            print(f"--- [TTSManager] Audio saved: {file_name} ---")
            return file_name
            
        except Exception as e:
            print(f"--- [TTSManager] Error during synthesis: {e} ---")
            raise
