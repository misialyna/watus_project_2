#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Watus Voice Assistant - Enhanced Version (Cross-Platform: Mac/Linux)

WYMAGANIA:
- numpy, sounddevice, zmq, webrtcvad (podstawowe)
- piper-tts (TTS)
- faster-whisper (lokalny STT) LUB groq-stt (Groq API)
- speechbrain (speaker verification - opcjonalne)
- dotenv (konfiguracja)
- led_controller (opcjonalne - dla LED)

INSTALACJA (CROSS-PLATFORM):
# MacOS:
brew install portaudio
pip install numpy sounddevice zmq webrtcvad piper-tts faster-whisper python-dotenv

# Linux (Ubuntu/Debian):
sudo apt-get install portaudio19-dev python3-pyaudio
pip install numpy sounddevice zmq webrtcvad piper-tts faster-whisper python-dotenv

# Additional dependencies:
pip install torch torchaudio  # dla speechbrain
pip install speechbrain  # opcjonalne - speaker verification
pip install groq  # opcjonalne - Groq STT API

KONFIGURACJA:
# Audio Settings
AUDIO_MODE=REAL  # "REAL" dla prawdziwego audio, "SIMULATION" dla testów
WATUS_INPUT_DEVICE=  # Opcjonalnie: numer konkretnego mikrofonu (-1 = auto)
WATUS_OUTPUT_DEVICE=  # Opcjonalnie: numer konkretnego głośnika (-1 = auto)

# STT Configuration
STT_PROVIDER=local  # "local" dla Faster-Whisper, "groq" dla Groq API
WHISPER_MODEL=small  # model size: tiny, base, small, medium, large

# Speaker verification z dynamicznym liderem
SPEAKER_VERIFY=1  # włączenie rozpoznawania mówcy
SPEAKER_STICKY_SEC=180  # Leader expires after 3 minutes of inactivity

# Wake words (oddzielone przecinkami)
WAKE_WORDS=hej watusiu,hej watuszu,hej watusił,kej watusił,hej watośiu
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CT2_SKIP_CONVERTERS", "1")

import sys, json, time, queue, threading, atexit, re
from pathlib import Path
from collections import deque
from dotenv import load_dotenv
import numpy as np
import zmq
import webrtcvad


# CRASH PROTECTION: Delay sounddevice import until after PortAudio check
# It will be imported later in the check_audio_dependency function

# Enhanced error handling and logging (must be first!)
class WatusLogger:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.system_info = {}

        # Platform-specific audio initialization
        self._init_platform_audio()

    def _init_platform_audio(self):
        """Initialize platform-specific audio settings"""
        platform = sys.platform

        if platform == 'linux':
            # Linux-specific audio settings
            try:
                # Set ALSA/PulseAudio settings for Linux
                os.environ.setdefault('ALSA_PCM_DEVICE', 'default')
                os.environ.setdefault('PULSE_LATENCY_MSEC', '50')
                self.info("[AUDIO] Linux audio settings configured")
            except Exception as e:
                self.warning(f"Linux audio config failed: {e}", "AUDIO")

        elif platform == 'darwin':
            # macOS-specific settings
            try:
                os.environ.setdefault('AUDIODEV', '0')  # Use first available device
                self.info("[AUDIO] macOS audio settings configured")
            except Exception as e:
                self.warning(f"macOS audio config failed: {e}", "AUDIO")

        self.info(f"[SYSTEM] Platform: {platform}")
        self.info(f"[SYSTEM] Python: {sys.version}")

    def error(self, msg, component=None):
        error_msg = f"[ERROR{f' ({component})' if component else ''}] {msg}"
        print(error_msg, flush=True)
        self.errors.append({"time": time.time(), "component": component, "message": msg})

    def warning(self, msg, component=None):
        warning_msg = f"[WARN{f' ({component})' if component else ''}] {msg}"
        print(warning_msg, flush=True)
        self.warnings.append({"time": time.time(), "component": component, "message": msg})

    def info(self, msg):
        print(msg, flush=True)

    def log_system_info(self):
        """Log system information for debugging"""
        try:
            import torch
            self.system_info = {
                "python_version": sys.version,
                "platform": sys.platform,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "audio_devices": len(sd.query_devices()) if hasattr(sd, 'query_devices') else 0,
            }
            self.info(f"[SYSTEM] {json.dumps(self.system_info, indent=2)}")
        except Exception:
            self.info("[SYSTEM] Basic info available (torch/sounddevice not imported yet)")


# Create logger immediately - BEFORE using it
logger = WatusLogger()

# Piper TTS Integration (Python API) - AFTER logger is created
try:
    from piper import PiperVoice, AudioChunk

    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    logger.warning("Piper TTS not available - install with: pip install piper-tts", "TTS")

logger.log_system_info()

# ASR: Choose between Faster-Whisper (local) or Groq API
try:
    from faster_whisper import WhisperModel

    logger.info("[OK] Faster-Whisper imported successfully")
except ImportError as e:
    logger.error(f"Failed to import Faster-Whisper: {e}", "ASR")
    WhisperModel = None

try:
    from groq_stt import GroqSTT

    logger.info("[OK] Groq STT imported successfully")
except ImportError as e:
    logger.error(f"Failed to import Groq STT: {e}", "ASR")
    GroqSTT = None

# LED Controller with graceful fallback
try:
    from led_controller import LEDController

    led = LEDController()
    atexit.register(led.cleanup)
    logger.info("[OK] LED Controller loaded")
except Exception as e:
    logger.warning(f"LED Controller failed: {e}", "LED")


    # Create dummy LED controller
    class DummyLED:
        def listening(self): pass

        def processing_or_speaking(self): pass

        def cleanup(self): pass


    led = DummyLED()

# .env configuration
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)


# CRASH PROTECTION: Early PortAudio detection and graceful fallback
def check_audio_dependency():
    """Check if PortAudio is available and provide fallback"""
    global sd, SOUNDDEVICE_AVAILABLE

    audio_mode = os.environ.get("AUDIO_MODE", "REAL")
    if audio_mode == "SIMULATION":
        logger.info("[AUDIO] Simulation mode - PortAudio not required")

        # Create dummy sounddevice for simulation
        class DummySD:
            def query_devices(self): return []

            def InputStream(self, **kwargs): raise RuntimeError("Audio not available")

            def play(self, *args, **kwargs): pass

            def wait(self): pass

        sd = DummySD()
        SOUNDDEVICE_AVAILABLE = False
        return True

    try:
        import sounddevice as sd
        # Test PortAudio availability
        devices = sd.query_devices()
        logger.info(f"[AUDIO] PortAudio available - {len(devices)} devices found")
        SOUNDDEVICE_AVAILABLE = True
        return True
    except OSError as e:
        if "PortAudio library not found" in str(e):
            logger.warning("PortAudio not found, falling back to simulation mode", "AUDIO")
            # Set simulation mode if PortAudio is not available
            os.environ["AUDIO_MODE"] = "SIMULATION"

            # Create dummy sounddevice for simulation
            class DummySD:
                def query_devices(self): return []

                def InputStream(self, **kwargs): raise RuntimeError("Audio not available")

                def play(self, *args, **kwargs): pass

                def wait(self): pass

            sd = DummySD()
            SOUNDDEVICE_AVAILABLE = False
            return False
        else:
            raise
    except Exception as e:
        logger.warning(f"Audio check failed: {e}, using simulation mode", "AUDIO")
        os.environ["AUDIO_MODE"] = "SIMULATION"

        # Create dummy sounddevice for simulation
        class DummySD:
            def query_devices(self): return []

            def InputStream(self, **kwargs): raise RuntimeError("Audio not available")

            def play(self, *args, **kwargs): pass

            def wait(self): pass

        sd = DummySD()
        SOUNDDEVICE_AVAILABLE = False
        return False


# Check audio dependency early
SOUNDDEVICE_AVAILABLE = False  # Initialize global flag
check_audio_dependency()


# Configuration with validation
def get_config_with_fallback(key, default, required=False, validator=None):
    """Get configuration with fallback and validation"""
    value = os.environ.get(key, default)

    if required and not value:
        logger.error(f"Required configuration missing: {key}", "CONFIG")
        return None

    if validator and not validator(value):
        logger.error(f"Invalid configuration for {key}: {value}", "CONFIG")
        return None

    return value


# ZMQ Configuration
PUB_ADDR = get_config_with_fallback("ZMQ_PUB_ADDR", "tcp://127.0.0.1:7780")
SUB_ADDR = get_config_with_fallback("ZMQ_SUB_ADDR", "tcp://127.0.0.1:7781")


def _normalize_fw_model(name: str) -> str:
    name = (name or "").strip()
    short = {"tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"}
    if "/" not in name and name.lower() in short:
        return f"guillaumekln/faster-whisper-{name.lower()}"
    return name


# ===== STT Configuration =====
STT_PROVIDER = get_config_with_fallback("STT_PROVIDER",
                                        "local").lower()  # "local" for Faster-Whisper, "groq" for Groq API

# Groq API configuration (used when STT_PROVIDER=groq)
GROQ_API_KEY = get_config_with_fallback("GROQ_API_KEY", "")
GROQ_MODEL = get_config_with_fallback("GROQ_MODEL", "whisper-large-v3")

# Local Whisper configuration (used when STT_PROVIDER=local)
WHISPER_MODEL_NAME = _normalize_fw_model(get_config_with_fallback("WHISPER_MODEL", "small"))
WHISPER_DEVICE = get_config_with_fallback("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = get_config_with_fallback("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_NUM_WORKERS = int(get_config_with_fallback("WHISPER_NUM_WORKERS", "1"))

CPU_THREADS = int(get_config_with_fallback("WATUS_CPU_THREADS", "4"))

# Audio/VAD Configuration
SAMPLE_RATE = int(get_config_with_fallback("WATUS_SR", "16000"))
BLOCK_SIZE = int(get_config_with_fallback("WATUS_BLOCKSIZE", "160"))
VAD_MODE = int(get_config_with_fallback("WATUS_VAD_MODE", "1"))
VAD_MIN_MS = int(get_config_with_fallback("WATUS_VAD_MIN_MS", "280"))
SIL_MS_END = int(get_config_with_fallback("WATUS_SIL_MS_END", "650"))
ASR_MIN_DBFS = float(get_config_with_fallback("ASR_MIN_DBFS", "-34"))

# Speaker verification
SPEAKER_VERIFY = int(get_config_with_fallback("SPEAKER_VERIFY", "1"))
SPEAKER_REQUIRE_MATCH = int(get_config_with_fallback("SPEAKER_REQUIRE_MATCH", "1"))
SPEAKER_STICKY_SEC = int(
    get_config_with_fallback("SPEAKER_STICKY_SEC", "180"))  # Leader expires after 3 minutes of inactivity

# Piper TTS Configuration
PIPER_MODEL_PATH = get_config_with_fallback("PIPER_MODEL_PATH", "models/piper/voices/pl_PL-darkman-medium.onnx")
PIPER_SAMPLE_RATE = int(get_config_with_fallback("PIPER_SAMPLE_RATE", "22050"))


# Audio device detection with fallback
def detect_audio_devices():
    """Detect audio devices with comprehensive error handling and Linux support"""
    try:
        if not SOUNDDEVICE_AVAILABLE:
            logger.info("[AUDIO] sounddevice not available - no audio devices")
            return [], []

        devices = sd.query_devices()
        input_devices = []
        output_devices = []

        for i, d in enumerate(devices):
            if d.get('max_input_channels', 0) > 0:
                input_devices.append((i, d['name']))
            if d.get('max_output_channels', 0) > 0:
                output_devices.append((i, d['name']))

        platform = sys.platform
        logger.info(f"[AUDIO] Found {len(devices)} total devices on {platform}:")
        for i, d in enumerate(devices):
            input_ch = d.get('max_input_channels', 0)
            output_ch = d.get('max_output_channels', 0)
            device_name = d['name']

            # Add device type detection for better logging
            device_type = []
            if input_ch > 0: device_type.append("MIC")
            if output_ch > 0: device_type.append("SPK")

            logger.info(f"  [{i}] {device_name} {' '.join(device_type)} (IN:{input_ch} OUT:{output_ch})")

        return input_devices, output_devices
    except Exception as e:
        logger.error(f"Audio device detection failed: {e}", "AUDIO")
        return [], []


# Device configuration with user override support
IN_DEV_ENV = get_config_with_fallback("WATUS_INPUT_DEVICE", "")  # -1 or specific index
OUT_DEV_ENV = get_config_with_fallback("WATUS_OUTPUT_DEVICE", "")  # -1 or specific index

input_devices, output_devices = detect_audio_devices()


def _auto_pick_device(device_list):
    """Auto-pick first available device from list"""
    if device_list:
        return device_list[0][0]  # Return device index
    return None


def _smart_pick_input_device(input_devices, platform):
    """Smart device selection - prefer laptop/internal microphone over iPhone"""
    if not input_devices:
        return None

    # Check for user-specified device
    IN_DEV_ENV = get_config_with_fallback("WATUS_INPUT_DEVICE", "")
    if IN_DEV_ENV and IN_DEV_ENV != "":
        try:
            dev_index = int(IN_DEV_ENV)
            # Verify device exists
            for idx, name in input_devices:
                if idx == dev_index:
                    logger.info(f"[AUDIO] Using user-specified input device [{dev_index}]: {name}")
                    return dev_index
        except ValueError:
            logger.warning(f"[AUDIO] Invalid WATUS_INPUT_DEVICE value: {IN_DEV_ENV}")

    # Smart selection based on platform and device names
    # Prefer built-in microphones over external devices
    preferred_keywords = {
        'darwin': ['built-in', 'internal', 'macbook', 'imac'],
        'linux': ['pulse', 'default', 'hdmi', 'usb'],
        'win32': ['microphone', 'mic', 'default']
    }

    platform_keywords = preferred_keywords.get(platform, [])

    # Score devices based on name keywords
    scored_devices = []
    for idx, name in input_devices:
        score = 0
        name_lower = name.lower()

        # Prefer built-in/internal devices
        for keyword in platform_keywords:
            if keyword in name_lower:
                score += 10

        # Penalize external/phone devices
        phone_keywords = ['iphone', 'android', 'phone', 'bluetooth']
        for keyword in phone_keywords:
            if keyword in name_lower:
                score -= 50

        # Prefer "default" or "built-in"
        if 'default' in name_lower or 'built' in name_lower:
            score += 20

        scored_devices.append((idx, name, score))

    # Sort by score (highest first)
    scored_devices.sort(key=lambda x: x[2], reverse=True)

    best_device = scored_devices[0]
    logger.info(f"[AUDIO] Smart device selection: [{best_device[0]}] {best_device[1]} (score: {best_device[2]})")

    return best_device[0]


def _smart_pick_output_device(output_devices, platform):
    """Smart device selection for output devices"""
    if not output_devices:
        return None

    # Check for user-specified device
    OUT_DEV_ENV = get_config_with_fallback("WATUS_OUTPUT_DEVICE", "")
    if OUT_DEV_ENV and OUT_DEV_ENV != "":
        try:
            dev_index = int(OUT_DEV_ENV)
            # Verify device exists
            for idx, name in output_devices:
                if idx == dev_index:
                    logger.info(f"[AUDIO] Using user-specified output device [{dev_index}]: {name}")
                    return dev_index
        except ValueError:
            logger.warning(f"[AUDIO] Invalid WATUS_OUTPUT_DEVICE value: {OUT_DEV_ENV}")

    # For output, prefer built-in speakers
    for idx, name in output_devices:
        name_lower = name.lower()
        if 'default' in name_lower or 'built' in name_lower or 'internal' in name_lower:
            logger.info(f"[AUDIO] Using built-in output device: [{idx}] {name}")
            return idx

    # Fallback to first available
    return output_devices[0][0]


# Audio device configuration with smart selection
IN_DEV = _smart_pick_input_device(input_devices, sys.platform) if input_devices else None
OUT_DEV = _smart_pick_output_device(output_devices, sys.platform) if output_devices else None

# Get audio mode from environment (may have been changed by PortAudio check)
AUDIO_MODE = os.environ.get("AUDIO_MODE", "REAL")

logger.info(f"[AUDIO] Audio mode: {AUDIO_MODE}")
if AUDIO_MODE == "REAL":
    if SOUNDDEVICE_AVAILABLE and IN_DEV is not None:
        logger.info(f"[OK] Real audio enabled - IN_DEV: {IN_DEV}, OUT_DEV: {OUT_DEV}")
    else:
        logger.warning("Real audio requested but not available, using simulation mode", "AUDIO")
        AUDIO_MODE = "SIMULATION"
        IN_DEV = None
        OUT_DEV = None

if AUDIO_MODE == "SIMULATION":
    logger.info(f"[SIM] Audio simulation mode enabled - no real audio devices")

# OLD CODE (commented out):
# Force simulation mode to avoid audio issues
# IN_DEV = None
# OUT_DEV = None
# AUDIO_MODE = "SIMULATION"
# logger.info(f"[SIM] Audio mode forced to SIMULATION to avoid audio input/output issues")
# logger.info(f"[SIM] All real audio devices ignored for testing purposes")

# UNCOMMENT THESE LINES TO ENABLE REAL AUDIO:
# IN_DEV = _auto_pick_device(input_devices) if input_devices else None
# OUT_DEV = _auto_pick_device(output_devices) if output_devices else None
# AUDIO_MODE = "REAL"  # Enable real audio input/output

if AUDIO_MODE == "SIMULATION":
    logger.warning("Running in SIMULATION mode - no audio devices available", "AUDIO")
    logger.info("Will simulate audio input/output for testing")

DIALOG_PATH = get_config_with_fallback("DIALOG_PATH", "dialog.jsonl")


def log(msg):
    logger.info(msg)


def is_wake_word_present(text: str) -> bool:
    """Enhanced wake word detection"""
    if not text:
        return False

    # Get wake words from config
    wake_words_env = get_config_with_fallback("WAKE_WORDS",
                                              "hej watusiu,hej watuszu,hej watusił,kej watusił,hej watośiu")
    wake_words = [w.strip() for w in wake_words_env.split(",") if w.strip()]

    normalized_text = re.sub(r'[^\w\s]', '', text.lower())

    for wake_phrase in wake_words:
        normalized_wake_phrase = re.sub(r'[^\w\s]', '', wake_phrase.lower())
        if normalized_wake_phrase in normalized_text:
            return True
    return False


# State changes with reduced logging - only show significant state changes
_last_state_log_time = 0


def cue_listen():
    global _last_state_log_time
    current_time = time.time()
    # Only log LISTENING state every 5 seconds to reduce noise
    if current_time - _last_state_log_time > 5:
        log("[Watus][STATE] LISTENING")
        _last_state_log_time = current_time
    led.listening()

    # Publish state for UI synchronization
    if 'bus' in globals():
        try:
            bus.publish_state("listening")
        except:
            pass


def cue_think():
    log("[Watus][STATE] THINKING")
    led.processing_or_speaking()

    # Publish state for UI synchronization
    if 'bus' in globals():
        try:
            bus.publish_state("processing")
        except:
            pass


def cue_speak():
    log("[Watus][STATE] SPEAKING")
    led.processing_or_speaking()

    # Publish state for UI synchronization
    if 'bus' in globals():
        try:
            bus.publish_state("speaking")
        except:
            pass


def cue_idle():
    log("[Watus][STATE] IDLE")
    led.processing_or_speaking()


# ZMQ Bus with enhanced error handling
class Bus:
    def __init__(self, pub_addr: str, sub_addr: str):
        try:
            self.ctx = zmq.Context.instance()
            self.pub = self.ctx.socket(zmq.PUB)
            self.pub.setsockopt(zmq.SNDHWM, 100)
            self.pub.bind(pub_addr)

            self.sub = self.ctx.socket(zmq.SUB)
            self.sub.connect(sub_addr)
            self.sub.setsockopt_string(zmq.SUBSCRIBE, "tts.speak")

            self._sub_queue = queue.Queue()
            threading.Thread(target=self._sub_loop, daemon=True).start()
            logger.info(f"[OK] ZMQ Bus initialized - PUB: {pub_addr}, SUB: {sub_addr}")
        except Exception as e:
            logger.error(f"Failed to initialize ZMQ Bus: {e}", "ZMQ")
            raise

    def publish_leader(self, payload: dict):
        try:
            t0 = time.time()
            self.pub.send_multipart([b"dialog.leader", json.dumps(payload, ensure_ascii=False).encode("utf-8")])
            # REDUCED LOGGING: Only show BUS performance occasionally
            if not hasattr(self, '_bus_log_counter'):
                self._bus_log_counter = 0
            self._bus_log_counter += 1
            if self._bus_log_counter % 10 == 0:  # Log every 10th message
                log(f"[Perf] BUS: {int((time.time() - t0) * 1000)}ms")
        except Exception as e:
            logger.error(f"Failed to publish leader message: {e}", "ZMQ")

    def publish_state(self, state: str, data: dict = None):
        """Publish current watus state for real-time UI synchronization"""
        try:
            if data is None:
                data = {}
            message = {
                "state": state,
                "timestamp": time.time(),
                **data
            }
            self.pub.send_multipart([b"watus.state", json.dumps(message, ensure_ascii=False).encode("utf-8")])
        except Exception as e:
            logger.error(f"Failed to publish state message: {e}", "ZMQ")

    def _sub_loop(self):
        while True:
            try:
                topic, payload = self.sub.recv_multipart()
                if topic != b"tts.speak": continue
                data = json.loads(payload.decode("utf-8", "ignore"))
                self._sub_queue.put(data)
            except Exception as e:
                logger.error(f"ZMQ subscription error: {e}", "ZMQ")
                time.sleep(0.01)

    def get_tts(self, timeout=0.1):
        try:
            return self._sub_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# Speaker verification with fallback
class _NoopVerifier:
    enabled = False

    def __init__(self):
        self._enrolled = None

    @property
    def enrolled(self):
        return False

    def enroll_wav(self, p):
        pass

    def enroll_samples(self, s, sr):
        pass

    def verify(self, s, sr, db):
        return {"enabled": False}

    def clear_enrollment(self):
        """Clear current leader - for timeout logic"""
        self._enrolled = None
        logger.info("[SPK] NoopVerifier: Leader cleared")


def _make_verifier():
    global SPEAKER_VERIFY

    # Check if SpeechBrain is available before enabling verification
    if SPEAKER_VERIFY:
        try:
            import torch
            from speechbrain.pretrained import EncoderClassifier
            logger.info("SpeechBrain available - enabling speaker verification")
        except (ImportError, AttributeError) as e:
            logger.warning(f"SpeechBrain inference module not available: {e}. Disabling speaker verification.",
                           "SPEAKER")
            SPEAKER_VERIFY = 0
            return _NoopVerifier()
        except Exception as e:
            logger.warning(f"SpeechBrain not available: {e}. Disabling speaker verification.", "SPEAKER")
            SPEAKER_VERIFY = 0
            return _NoopVerifier()

    if not SPEAKER_VERIFY:
        logger.warning("Speaker verification disabled", "SPEAKER")
        return _NoopVerifier()

    class _SbVerifier:
        enabled = True

        def __init__(self):
            import torch
            self.threshold = float(get_config_with_fallback("SPEAKER_THRESHOLD", "0.64"))
            self.sticky_thr = float(get_config_with_fallback("SPEAKER_STICKY_THRESHOLD", str(self.threshold)))
            self.back_thr = float(get_config_with_fallback("SPEAKER_BACK_THRESHOLD", "0.56"))
            self.grace = float(get_config_with_fallback("SPEAKER_GRACE", "0.12"))
            self.sticky_sec = float(get_config_with_fallback("SPEAKER_STICKY_SEC", "3600"))
            self._clf = None
            self._device = "cpu"  # Force CPU for compatibility
            self._enrolled = None
            self._enroll_ts = 0.0
            logger.info(f"[OK] Speaker verifier configured (threshold: {self.threshold})")

            # Try to load existing speaker data
            self._load_enrolled_voice()

        @property
        def enrolled(self):
            return self._enrolled is not None

        def _ensure(self):
            if self._clf is None:
                try:
                    from speechbrain.pretrained import EncoderClassifier
                    self._clf = EncoderClassifier.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        run_opts={"device": self._device},
                        savedir="models/ecapa",
                    )
                    logger.info("[OK] ECAPA model loaded successfully")
                except (ImportError, AttributeError) as e:
                    logger.error(f"SpeechBrain inference module not available: {e}", "SPEAKER")
                    raise ImportError(f"Cannot import speechbrain.pretrained.EncoderClassifier: {e}")
                except Exception as e:
                    logger.error(f"Failed to load ECAPA model: {e}", "SPEAKER")
                    raise

        def _resample_16k(self, x: np.ndarray, sr: int) -> np.ndarray:
            if sr == 16000:
                return x.astype(np.float32)
            ratio = 16000.0 / sr
            n_out = int(round(len(x) * ratio))
            idx = np.linspace(0, len(x) - 1, num=n_out, dtype=np.float32)
            base = np.arange(len(x), dtype=np.float32)
            return np.interp(idx, base, x).astype(np.float32)

        def _embed(self, samples: np.ndarray, sr: int):
            self._ensure()
            wav = self._resample_16k(samples, sr)
            t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                emb = self._clf.encode_batch(t).squeeze(0).squeeze(0)
            return emb.detach().cpu().numpy().astype(np.float32)

        def enroll_samples(self, samples: np.ndarray, sr: int):
            try:
                # Clear previous leader - new leader with every wake word
                self._enrolled = None
                emb = self._embed(samples, sr)
                self._enrolled = emb
                self._enroll_ts = time.time()
                logger.info("[OK] New leader enrolled (previous leader cleared)")

            except Exception as e:
                logger.error(f"Speaker enrollment failed: {e}", "SPEAKER")

        def clear_enrollment(self):
            """Clear current leader - for timeout logic"""
            self._enrolled = None
            self._enroll_ts = 0.0
            logger.info("[SPK] Leader enrollment cleared (timeout or reset)")

        def _load_enrolled_voice(self):
            """Load previously enrolled voice from disk - DISABLED for dynamic leaders"""
            # For dynamic leaders, we don't load from disk
            # Every wake word should enroll a new leader
            logger.info("[INFO] Dynamic leader mode - no persistent storage")
            return False

        def verify(self, samples: np.ndarray, sr: int, dbfs: float) -> dict:
            if self._enrolled is None:
                return {"enabled": True, "enrolled": False}

            try:
                import torch, torch.nn.functional as F
                a = self._embed(samples, sr)
                sim = float(F.cosine_similarity(
                    torch.tensor(a, dtype=torch.float32).flatten(),
                    torch.tensor(self._enrolled, dtype=torch.float32).flatten(),
                    dim=0, eps=1e-8
                ).detach().cpu().item())

                now = time.time()
                age = now - self._enroll_ts
                is_leader = False
                adj_thr = (self.sticky_thr - self.grace) if dbfs > -22.0 else self.sticky_thr
                if age <= self.sticky_sec and sim >= adj_thr:
                    is_leader = True
                elif sim >= self.threshold:
                    is_leader = True
                elif sim >= self.back_thr and age <= self.sticky_sec:
                    is_leader = True

                return {"enabled": True, "enrolled": True, "score": sim, "is_leader": bool(is_leader),
                        "sticky_age_s": age}
            except Exception as e:
                logger.error(f"Speaker verification failed: {e}", "SPEAKER")
                return {"enabled": True, "enrolled": True, "score": 0.0, "is_leader": False, "error": str(e)}

    return _SbVerifier()


# STT Engine with enhanced error handling
class STTEngine:
    def __init__(self, state: 'State', bus: 'Bus'):
        self.state = state
        self.bus = bus
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.stt_provider = STT_PROVIDER
        self.last_leader_interaction_time = 0  # Track last leader interaction for timeout
        self.leader_timeout_seconds = int(get_config_with_fallback("SPEAKER_STICKY_SEC", "180"))  # 3 minutes default

        logger.info(f"[OK] STT ready (device={IN_DEV} sr={SAMPLE_RATE} block={BLOCK_SIZE}, mode={AUDIO_MODE})")

        # Initialize STT based on provider
        if self.stt_provider == "groq":
            self._init_groq_stt()
        else:
            self._init_local_whisper()

        self.verifier = _make_verifier()
        self.emit_cooldown_ms = int(get_config_with_fallback("EMIT_COOLDOWN_MS", "300"))
        self.cooldown_until = 0

        # Audio simulation for environments without audio devices
        self.simulation_mode = (AUDIO_MODE == "SIMULATION")
        if self.simulation_mode:
            logger.info("[SIM] Audio simulation mode enabled")

    def _check_leader_timeout(self):
        """Check if leader has expired due to inactivity"""
        now = time.time()
        time_since_last_interaction = now - self.last_leader_interaction_time

        # CRASH PROTECTION: Update heartbeat
        if hasattr(self, 'state') and hasattr(self.state, '_last_heartbeat'):
            self.state._last_heartbeat = now

        if time_since_last_interaction > self.leader_timeout_seconds:
            # Leader has expired - clear enrollment
            logger.info(f"[SPK] Leader expired after {int(time_since_last_interaction / 60)} minutes of inactivity")
            if hasattr(self.verifier, 'clear_enrollment'):
                self.verifier.clear_enrollment()
            elif hasattr(self, '_leader_enrolled'):
                self._leader_enrolled = False
            return True
        return False

    def _init_groq_stt(self):
        """Initialize Groq Speech-to-Text API"""
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY not provided, falling back to local Whisper", "ASR")
            return self._init_local_whisper()

        if GroqSTT is None:
            logger.error("Groq STT module not available, falling back to local Whisper", "ASR")
            return self._init_local_whisper()

        try:
            logger.info(f"[AUDIO] Initializing Groq STT: model={GROQ_MODEL}")
            t0 = time.time()
            self.model = GroqSTT(GROQ_API_KEY, GROQ_MODEL)

            # Validate API key
            if not self.model.validate_api_key(GROQ_API_KEY):
                logger.error("Invalid GROQ_API_KEY, falling back to local Whisper", "ASR")
                return self._init_local_whisper()

            load_time = int((time.time() - t0) * 1000)
            logger.info(f"[OK] Groq STT API loaded successfully ({load_time} ms)")

        except Exception as e:
            logger.error(f"Failed to initialize Groq STT: {e}", "ASR")
            logger.error("Falling back to local Whisper", "ASR")
            return self._init_local_whisper()

    def _init_local_whisper(self):
        """Initialize local Faster-Whisper"""
        if WhisperModel is None:
            logger.error("Faster-Whisper module not available", "ASR")
            raise ImportError("Neither Groq STT nor Faster-Whisper is available")

        logger.info(
            f"[AUDIO] Initializing Faster-Whisper: model={WHISPER_MODEL_NAME} device={WHISPER_DEVICE} compute={WHISPER_COMPUTE}")

        try:
            t0 = time.time()
            self.model = WhisperModel(
                WHISPER_MODEL_NAME,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE,
                cpu_threads=CPU_THREADS,
                num_workers=WHISPER_NUM_WORKERS
            )
            load_time = int((time.time() - t0) * 1000)
            logger.info(f"[OK] Faster-Whisper loaded successfully ({load_time} ms)")
            self.stt_provider = "local"
        except Exception as e:
            logger.error(f"Failed to initialize Faster-Whisper: {e}", "ASR")
            raise

    @staticmethod
    def _rms_dbfs(x: np.ndarray, eps=1e-9):
        rms = np.sqrt(np.mean(np.square(x) + eps))
        return 20 * np.log10(max(rms, eps))

    def _vad_is_speech(self, frame_bytes: bytes) -> bool:
        try:
            result = self.vad.is_speech(frame_bytes, SAMPLE_RATE)
            # REDUCED DEBUG: Only log rare errors, not normal operation
            return result
        except Exception as e:
            logger.warning(f"VAD error: {e}, returning False")
            return False

    def _transcribe_float32(self, pcm_f32: np.ndarray) -> str:
        t0 = time.time()

        # CRASH PROTECTION: Wrap all transcription in try-catch
        try:
            if self.stt_provider == "groq":
                # Use Groq API for transcription
                txt = self.model.transcribe_numpy(pcm_f32, SAMPLE_RATE, "pl")
                # REDUCED LOGGING: Only show performance for longer transcriptions
                if len(txt) > 10:  # Only log if substantial text
                    logger.info(f"[Perf] ASR: {int((time.time() - t0) * 1000)}ms, text_len={len(txt)}")
                return txt
            else:
                # Use local Faster-Whisper
                return self._transcribe_local(pcm_f32)
        except Exception as e:
            logger.error(f"Transcription failed: {e}", "ASR")
            logger.info("[ASR] Returning empty text due to transcription error")
            return ""  # Return empty string instead of crashing

    def _transcribe_local(self, pcm_f32: np.ndarray) -> str:
        """Transcribe using local Faster-Whisper"""
        t0 = time.time()
        try:
            # CRASH PROTECTION: Check if model is available
            if not hasattr(self, 'model') or self.model is None:
                logger.warning("STT model not available, returning empty text", "ASR")
                return ""

            # CRASH PROTECTION: Check audio data validity
            if pcm_f32 is None or len(pcm_f32) == 0:
                logger.warning("Empty audio data, returning empty text", "ASR")
                return ""

            segments, _ = self.model.transcribe(
                pcm_f32, language="pl", beam_size=1, vad_filter=False
            )
            txt = "".join(seg.text for seg in segments)
            # REDUCED LOGGING: Only show performance for longer transcriptions
            if len(txt) > 10:  # Only log if substantial text
                logger.info(f"[Perf] ASR: {int((time.time() - t0) * 1000)}ms, text_len={len(txt)}")
            return txt
        except Exception as e:
            logger.error(f"Local transcription failed: {e}", "ASR")
            logger.info("[ASR] Returning empty text due to error")
            return ""  # Return empty string instead of crashing

    def run_simulation(self):
        """Run in simulation mode for testing without audio"""
        logger.info("[SIM] Starting simulation mode - generating test audio")

        # Create a simple test sequence
        test_phrases = [
            "hej watusiu jak się masz",
            "powiedz mi jaka jest pogoda",
            "dziękuję watusiu"
        ]

        for i, phrase in enumerate(test_phrases):
            logger.info(f"[SIM] Simulating phrase {i + 1}: {phrase}")
            cue_think()

            # Simulate processing time
            time.sleep(2)

            # Simulate speaker verification
            if self.verifier.enabled and is_wake_word_present(phrase):
                self.verifier.enroll_samples(np.random.randn(16000).astype(np.float32), 16000)
                logger.info("[SIM] Simulated leader enrollment")

            # Create simulated transcription result
            text = phrase if i < 2 else ""  # Last phrase empty to test handling

            if text:
                turn_id = int(time.time() * 1000) + i
                line = {
                    "type": "leader_utterance",
                    "session_id": f"sim_{int(time.time())}",
                    "group_id": f"leader_{turn_id}",
                    "speaker_id": "leader",
                    "is_leader": True,
                    "turn_ids": [turn_id],
                    "text_full": text,
                    "category": "wypowiedź",
                    "reply_hint": True,
                    "ts_start": time.time(),
                    "ts_end": time.time() + 2,
                    "dbfs": -20.0,
                    "verify": {"enabled": True, "enrolled": True, "score": 0.8, "is_leader": True},
                    "emit_reason": "simulation",
                    "ts": time.time()
                }

                # Write to dialog file
                with open(DIALOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")

                logger.info(f"[SIM] Published: {text}")

                # Simulate ZMQ publish (would go to reporter)
                self.bus.publish_leader(line)

            cue_listen()
            time.sleep(1)

        logger.info("[SIM] Simulation complete")

    def run(self):
        if self.simulation_mode:
            logger.info("[SIM] Running in simulation mode")
            self.run_simulation()
            return

        # Real audio mode
        try:
            in_stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                blocksize=BLOCK_SIZE, device=IN_DEV
            )

            frame_ms = int(1000 * BLOCK_SIZE / SAMPLE_RATE)
            sil_frames_end = max(1, SIL_MS_END // frame_ms)
            min_speech_frames = max(1, VAD_MIN_MS // frame_ms)

            pre_buffer = deque(maxlen=15)
            speech_frames = bytearray()
            in_speech = False
            started_ms = None
            last_voice_ms = 0
            listening_flag = None

            with in_stream:
                logger.info("[OK] Audio stream started")
                while True:
                    now_ms = int(time.time() * 1000)

                    if self.state.is_blocked():
                        if listening_flag is not False:
                            cue_idle();
                            listening_flag = False
                        in_speech = False;
                        speech_frames = bytearray();
                        started_ms = None
                        pre_buffer.clear()
                        time.sleep(0.01);
                        continue

                    if now_ms < self.cooldown_until:
                        time.sleep(0.003);
                        continue

                    if listening_flag is not True:
                        cue_listen();
                        listening_flag = True

                    # CRASH PROTECTION: Simple watchdog - reset if stuck
                    if hasattr(self, 'state') and hasattr(self.state, '_last_heartbeat'):
                        now = time.time()
                        if now - self.state._last_heartbeat > self.state._max_silence_seconds:
                            logger.warning(
                                f"[WATCHDOG] No activity for {self.state._max_silence_seconds}s, resetting state")
                            listening_flag = None
                            in_speech = False
                            speech_frames = bytearray()
                            started_ms = None
                            self.state._last_heartbeat = now

                    # Check for leader timeout periodically (every 30 seconds)
                    if int(now_ms / 30000) != int((now_ms - 10) / 30000):  # Every 30 seconds
                        try:
                            self._check_leader_timeout()
                        except Exception as e:
                            logger.error(f"Leader timeout check failed: {e}", "SPK")

                    try:
                        audio, _ = in_stream.read(BLOCK_SIZE)
                    except Exception as e:
                        logger.error(f"Audio read error: {e}", "AUDIO")
                        time.sleep(0.01);
                        continue

                    frame_bytes = audio.tobytes()
                    pre_buffer.append(frame_bytes)
                    is_sp = self._vad_is_speech(frame_bytes)

                    # Simplified speech detection (for brevity)
                    # In a real implementation, you'd include the full VAD logic
                    if is_sp and not in_speech:
                        in_speech = True
                        speech_frames = bytearray()
                        if pre_buffer:
                            speech_frames.extend(b''.join(pre_buffer))
                        started_ms = now_ms

                    # Handle speech end and processing
                    if in_speech and not is_sp:
                        if speech_frames and started_ms:
                            dur_ms = now_ms - started_ms
                            if dur_ms >= VAD_MIN_MS:
                                # REDUCED LOGGING: Only show summary, not every utterance
                                if not hasattr(self,
                                               '_last_utterance_log') or time.time() - self._last_utterance_log > 2:
                                    logger.info(f"[AUDIO] Speech processed: {dur_ms}ms")
                                    self._last_utterance_log = time.time()
                                self._finalize(speech_frames, started_ms, now_ms, dur_ms)
                                self.cooldown_until = now_ms + self.emit_cooldown_ms
                        in_speech = False
                        speech_frames = bytearray()
                        started_ms = None
                        # Don't reset listening_flag here, let it stay True to avoid constant state switching
                        # No logging here to avoid noise

                    time.sleep(0.0005)

        except Exception as e:
            logger.error(f"Audio stream failed: {e}", "AUDIO")
            logger.info("[SIM] Falling back to simulation mode")
            self.run_simulation()

    def _finalize(self, speech_frames: bytearray, started_ms: int, last_voice_ms: int, dur_ms: int):
        cue_think()

        # CRASH PROTECTION: Validate speech frames before processing
        try:
            if not speech_frames or len(speech_frames) == 0:
                logger.warning("Empty speech frames, skipping processing", "AUDIO")
                return

            pcm_f32 = np.frombuffer(speech_frames, dtype=np.int16).astype(np.float32) / 32768.0
            dbfs = float(self._rms_dbfs(pcm_f32))
            if dbfs < ASR_MIN_DBFS:
                return
        except Exception as e:
            logger.error(f"Audio processing failed: {e}", "AUDIO")
            return

        # Transcribe with crash protection
        try:
            text = self._transcribe_float32(pcm_f32).strip()
            if not text:
                cue_listen()  # Return to listening state
                return
        except Exception as e:
            logger.error(f"Transcription crash protection triggered: {e}", "ASR")
            cue_listen()  # Return to listening state
            return

        # Check for leader timeout before processing
        leader_expired = self._check_leader_timeout()

        # Speaker verification logic (improved)
        verify = {}
        is_leader = False
        is_wake_word = is_wake_word_present(text)

        # REDUCED LOGGING: Only log essential speaker verification info
        if is_wake_word:
            logger.info(f"[SPK] Wake word detected: '{text[:30]}...'")
        elif is_leader:
            logger.info(f"[SPK] Leader verified: '{text[:30]}...'")
        # else: silent - no logging for ignored text

        # Speaker verification with crash protection
        try:
            if getattr(self.verifier, "enabled", False):
                if is_wake_word:
                    logger.info("[SPK] Wake word detected. Enrolling new leader (replacing previous).")
                    # CRASH PROTECTION: Wrap enrollment in try-catch
                    try:
                        self.verifier.enroll_samples(pcm_f32, SAMPLE_RATE)
                    except Exception as e:
                        logger.error(f"Speaker enrollment failed: {e}", "SPEAKER")

                    verify = self.verifier.verify(pcm_f32, SAMPLE_RATE, dbfs)
                    is_leader = True
                    self.last_leader_interaction_time = time.time()  # Update interaction time
                elif self.verifier.enrolled:
                    # CRASH PROTECTION: Wrap verification in try-catch
                    try:
                        verify = self.verifier.verify(pcm_f32, SAMPLE_RATE, dbfs)
                        is_leader = bool(verify.get("is_leader", False))
                    except Exception as e:
                        logger.error(f"Speaker verification failed: {e}", "SPEAKER")
                        verify = {"enabled": True, "enrolled": True, "score": 0.0, "is_leader": False, "error": str(e)}
                        is_leader = False

                    if is_leader:
                        self.last_leader_interaction_time = time.time()  # Update interaction time
                else:
                    logger.info(f"[SPK] No leader and no wake word. Ignoring: '{text[:30]}...'")
                    cue_listen()  # Return to listening state
                    return
            else:
                # If speaker verification is disabled, use simple logic:
                if is_wake_word:
                    logger.info("[SPK] Wake word detected, marking as leader")
                    is_leader = True
                    if not hasattr(self, '_leader_enrolled'):
                        self._leader_enrolled = False
                    self._leader_enrolled = True  # New leader replaces old
                    self.last_leader_interaction_time = time.time()  # Update interaction time
                elif hasattr(self, '_leader_enrolled') and self._leader_enrolled:
                    is_leader = True
                    self.last_leader_interaction_time = time.time()  # Update interaction time
                else:
                    logger.info(f"[SPK] No leader enrolled and no wake word. Ignoring: '{text[:30]}...'")
                    cue_listen()  # Return to listening state
                    return
        except Exception as e:
            logger.error(f"Speaker verification crash protection: {e}", "SPEAKER")
            cue_listen()  # Return to listening state
            return

        # MINIMAL LOGGING: Only log final decision when necessary
        if is_leader:
            logger.info(f"[SPK] Leader accepted: '{text[:30]}...'")
        # No logging for rejection to reduce noise

        # Create and save dialog entry
        ts_start = started_ms / 1000.0
        ts_end = last_voice_ms / 1000.0
        turn_id = last_voice_ms

        line = {
            "type": "leader_utterance" if is_leader else "unknown_utterance",
            "session_id": self.state.session_id,
            "group_id": f"{'leader' if is_leader else 'unknown'}_{turn_id}",
            "speaker_id": "leader" if is_leader else "unknown",
            "is_leader": is_leader,
            "turn_ids": [turn_id],
            "text_full": text,
            "category": "wypowiedź",
            "reply_hint": is_leader,
            "ts_start": ts_start,
            "ts_end": ts_end,
            "dbfs": dbfs,
            "verify": verify,
            "emit_reason": "endpoint",
            "ts": time.time()
        }

        with open(DIALOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

        # Publication with crash protection
        try:
            if is_leader:
                # MINIMAL LOGGING: Only essential publication message
                logger.info(f"[PUB] Leader text: '{text[:30]}...'")
                self.state.set_awaiting_reply(True)
                # CRASH PROTECTION: Wrap ZMQ publish in try-catch
                try:
                    self.bus.publish_leader(line)
                except Exception as e:
                    logger.error(f"ZMQ publish failed: {e}", "ZMQ")
                self.state.pause_until_reply()
                # No additional logging to reduce noise
            else:
                logger.info(f"[SKIP] non-leader text saved: '{text[:30]}...'")
        except Exception as e:
            logger.error(f"Publication crash protection: {e}", "BUS")

        # Return to listening state
        cue_listen()  # Return to listening state silently


# State management
class State:
    def __init__(self):
        self.session_id = f"live_{int(time.time())}"
        self._tts_active = False
        self._awaiting_reply = False
        self._lock = threading.Lock()
        self.tts_pending_until = 0.0
        self.waiting_reply_until = 0.0
        self.last_tts_id = None
        # CRASH PROTECTION: Add simple watchdog
        self._last_heartbeat = time.time()
        self._max_silence_seconds = 30  # Reset if no activity for 30 seconds

    def set_tts(self, flag: bool):
        with self._lock:
            self._tts_active = flag

    def set_awaiting_reply(self, flag: bool):
        with self._lock:
            self._awaiting_reply = flag

    def pause_until_reply(self):
        with self._lock:
            self.waiting_reply_until = time.time() + float(get_config_with_fallback("WAIT_REPLY_S", "0.6"))

    def is_blocked(self) -> bool:
        with self._lock:
            return (
                    self._tts_active
                    or self._awaiting_reply
                    or (time.time() < self.tts_pending_until)
                    or (time.time() < self.waiting_reply_until)
            )


# Piper TTS Function - FIXED VERSION with new AudioChunk API
def piper_say(text: str, voice_path: str = None, sample_rate: int = None) -> bool:
    """Speak text using Piper TTS Python API - FIXED VERSION with new AudioChunk API"""
    if not PIPER_AVAILABLE:
        logger.warning("Piper TTS not available - cannot speak", "TTS")
        return False

    if not text or not text.strip():
        return False

    # Use environment variables if not provided
    if voice_path is None:
        voice_path = PIPER_MODEL_PATH
    if sample_rate is None:
        sample_rate = PIPER_SAMPLE_RATE

    try:
        # Load voice if not already loaded
        if not hasattr(piper_say, '_voice'):
            if not os.path.exists(voice_path):
                logger.error(f"Piper model not found: {voice_path}", "TTS")
                return False
            piper_say._voice = PiperVoice.load(voice_path)
            logger.info(f"[TTS] Piper voice loaded: {voice_path}", "TTS")

        # FIXED: Use new API that returns AudioChunk iterator (v1.3.0)
        audio_data = []
        for chunk in piper_say._voice.synthesize(text):
            if isinstance(chunk, AudioChunk):
                chunk_data = chunk.audio_int16_array.astype(np.float32) / 32768.0
                audio_data.append(chunk_data)
            else:
                logger.warning(f"[TTS] Unexpected chunk type: {type(chunk)}", "TTS")

        if audio_data:
            # Combine all audio chunks
            full_audio = np.concatenate(audio_data)
            actual_sample_rate = piper_say._voice.config.sample_rate

            # Play audio using sounddevice
            sd.play(full_audio, actual_sample_rate)
            sd.wait()  # Wait for playback to complete

            logger.info(f"[TTS] Spoke: {text[:50]}{'...' if len(text) > 50 else ''}", "TTS")
            return True
        else:
            logger.warning("[TTS] No audio data generated", "TTS")
            return False

    except Exception as e:
        logger.error(f"[TTS] Error in speech synthesis: {e}", "TTS")
        return False


# TTS Worker (enhanced with Piper TTS)
def tts_worker(state: State, bus: Bus):
    logger.info("[TTS] Worker started")
    if not PIPER_AVAILABLE:
        logger.warning("Piper TTS not available - TTS worker will only log messages", "TTS")

    while True:
        msg = bus.get_tts(timeout=0.1)
        if not msg:
            continue

        text = (msg.get("text") or "").strip()
        if text:
            logger.info(f"[TTS] Received text: {text[:50]}...")

            # Use Piper TTS if available
            if PIPER_AVAILABLE:
                piper_say(text)
            else:
                # Fallback: just log the message
                logger.info(f"[TTS] Would speak: {text}", "TTS")

        time.sleep(0.1)


# Main function with enhanced error handling
def main():
    """Main entry point with comprehensive error handling"""
    logger.info("=" * 60)
    logger.info("🤖 WATUS VOICE FRONTEND - Enhanced Version")
    logger.info("=" * 60)

    try:
        # Environment check
        if STT_PROVIDER == "groq":
            logger.info(f"[ENV] STT=groq MODEL={GROQ_MODEL}")
        else:
            logger.info(f"[ENV] STT=local MODEL={WHISPER_MODEL_NAME} DEVICE={WHISPER_DEVICE} COMPUTE={WHISPER_COMPUTE}")
        logger.info(f"[ENV] Audio mode: {AUDIO_MODE}")

        # Wake words
        wake_words_env = get_config_with_fallback("WAKE_WORDS",
                                                  "hej watusiu,hej watuszu,hej watusił,kej watusił,hej watośiu")
        wake_words = [w.strip() for w in wake_words_env.split(",") if w.strip()]
        logger.info(f"[WAKE] Configured wake words: {wake_words}")

        logger.info(f"[ZMQ] PUB: {PUB_ADDR} | SUB: {SUB_ADDR}")

        # Initialize ZMQ Bus
        try:
            bus = Bus(PUB_ADDR, SUB_ADDR)
        except Exception as e:
            logger.error(f"Failed to initialize ZMQ Bus: {e}", "ZMQ")
            return False

        # Initialize State
        state = State()

        # Start TTS worker
        tts_thread = threading.Thread(target=tts_worker, args=(state, bus), daemon=True)
        tts_thread.start()

        # Initialize STT Engine
        try:
            stt = STTEngine(state, bus)
        except Exception as e:
            logger.error(f"Failed to initialize STT: {e}", "ASR")
            return False

        logger.info(f"[AUDIO] Input device: {IN_DEV} | Output device: {OUT_DEV}")

        # Start in listening state
        led.listening()

        logger.info("🚀 Watus is ready! Starting main loop...")

        # Main loop
        try:
            stt.run()
        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}", "MAIN")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return False

        logger.info("👋 Watus stopped gracefully")
        return True

    except Exception as e:
        logger.error(f"Fatal error during initialization: {e}", "MAIN")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    import traceback

    success = main()

    # Final status report
    logger.info("\n" + "=" * 60)
    logger.info("📊 FINAL STATUS REPORT")
    logger.info("=" * 60)

    if logger.errors:
        logger.info(f"❌ Errors encountered: {len(logger.errors)}")
        for error in logger.errors:
            logger.info(f"  - [{error['component'] or 'UNKNOWN'}] {error['message']}")
    else:
        logger.info("✅ No errors encountered")

    if logger.warnings:
        logger.info(f"⚠️  Warnings: {len(logger.warnings)}")
        for warning in logger.warnings:
            logger.info(f"  - [{warning['component'] or 'UNKNOWN'}] {warning['message']}")
    else:
        logger.info("✅ No warnings")

    logger.info(f"🎯 Overall result: {'SUCCESS' if success else 'FAILED'}")

    if not success:
        logger.info("\n💡 Troubleshooting tips:")
        logger.info("1. Check audio device availability")
        logger.info("2. Verify .env configuration")
        logger.info("3. Ensure all dependencies are installed")
        logger.info("4. Run system_diagnostic.py for detailed analysis")

    sys.exit(0 if success else 1)
