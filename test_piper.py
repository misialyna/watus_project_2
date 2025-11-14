#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Piper TTS Synthesis
"""

from piper import PiperVoice, AudioChunk
import tempfile
import os
import numpy as np
import wave


def test_piper_synthesis():
    """Test Piper TTS synthesis with proper API usage"""

    # Test ładowania modelu
    model_path = 'models/piper/voices/pl_PL-darkman-medium.onnx'
    try:
        voice = PiperVoice.load(model_path)
        print('✅ Model załadowany')
    except Exception as e:
        print(f'❌ Błąd ładowania modelu: {e}')
        return False

    # Test syntezy - corrected method
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            test_text = 'Test polskiego głosu'
            chunks = voice.synthesize(test_text)
            audio_data = []

            for chunk in chunks:
                if isinstance(chunk, AudioChunk):
                    # Use audio_int16_array for the Int16 audio data
                    audio_data.append(chunk.audio_int16_array)
                else:
                    print(f'❌ Nieoczekiwany typ chunk: {type(chunk)}')
                    return False

            if audio_data:
                full_audio = np.concatenate(audio_data)
                sample_rate = voice.config.sample_rate

                # Write to WAV file
                with wave.open(tmp.name, 'wb') as f:
                    f.setnchannels(1)          # Mono
                    f.setsampwidth(2)           # 16-bit
                    f.setframerate(sample_rate) # Sample rate from config
                    f.writeframes(full_audio.tobytes())

                # Check result
                if os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 1000:
                    print(f'✅ Synteza audio działa! Plik: {tmp.name} (rozmiar: {os.path.getsize(tmp.name)} bajtów)')
                    print(f'   Tekst: "{test_text}" (sample rate: {sample_rate} Hz)')
                    print('   Możesz odtworzyć plik audio, aby sprawdzić jakość syntezy')
                    # Don't delete the temp file so user can listen
                    return True
                else:
                    print('❌ Błąd syntezy - plik jest pusty lub zbyt mały')
                    os.unlink(tmp.name) if os.path.exists(tmp.name) else None
                    return False
            else:
                print('❌ Brak audio data')
                return False

    except Exception as e:
        print(f'❌ Błąd syntezy: {e}')
        return False


if __name__ == "__main__":
    print("=== Test Piper TTS ===")
    success = test_piper_synthesis()
    print("=== Wyniki testu ===")
    print(f"{'✅ SUKCES' if success else '❌ BŁĄD'}")
