"""Speech-to-text transcription service using WhisperX.

Adapts the interview-transcription logic into a reusable class for
use inside the Foldback Service. Handles model lifecycle, Windows
DLL path fixes, ffmpeg cache staging, and the torch.load
weights_only=False patch required by pyannote checkpoints.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Windows DLL path fixes (must run before whisperx is imported)
# ---------------------------------------------------------------------------
if os.name == "nt":
    import site

    for _sp in site.getsitepackages():
        for _sub in ("nvidia/cudnn/bin", "nvidia/cublas/bin", "nvidia/cuda_nvrtc/bin"):
            _dir = os.path.join(_sp, *_sub.split("/"))
            if os.path.isdir(_dir):
                os.add_dll_directory(_dir)
                os.environ["PATH"] = _dir + os.pathsep + os.environ["PATH"]

# whisperx.load_audio shells out to a subprocess named exactly "ffmpeg". The
# static binary shipped by imageio-ffmpeg has a versioned filename, so we stage
# a copy named ffmpeg.exe in a project-local cache dir and prepend that dir to
# PATH.
import imageio_ffmpeg  # noqa: E402

_ffmpeg_src = Path(imageio_ffmpeg.get_ffmpeg_exe())
_ffmpeg_cache = Path(__file__).parent.parent / ".cache" / "ffmpeg"
_ffmpeg_cache.mkdir(parents=True, exist_ok=True)
_ffmpeg_dst = _ffmpeg_cache / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
if not _ffmpeg_dst.exists() or _ffmpeg_dst.stat().st_size != _ffmpeg_src.stat().st_size:
    shutil.copy2(_ffmpeg_src, _ffmpeg_dst)
os.environ["PATH"] = str(_ffmpeg_cache) + os.pathsep + os.environ["PATH"]

# ---------------------------------------------------------------------------
# PyTorch 2.6 torch.load weights_only patch (must run before whisperx import)
# ---------------------------------------------------------------------------
_orig_torch_load = torch.load


def _torch_load_weights_only_false(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false

import whisperx  # noqa: E402

logger = logging.getLogger(__name__)


class TranscriptionService:
    """WhisperX-based transcription with optional alignment and speaker diarisation."""

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str | None = None,
        compute_type: str = "float16",
        language: str | None = "en",
        batch_size: int = 16,
        diarise: bool = False,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type
        self.language = language
        self.batch_size = batch_size
        self.diarise = diarise
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

        self._asr_model: object | None = None
        self._diarise_pipeline: object | None = None

    def load_model(self) -> None:
        """Load the WhisperX model, alignment model, and optional diarisation pipeline."""
        if self._asr_model is not None:
            logger.info("ASR model already loaded; skipping reload.")
            return

        logger.info("Loading ASR model %s on %s (compute_type=%s)", self.model_name, self.device, self.compute_type)
        self._asr_model = whisperx.load_model(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            language=self.language,
        )

        if self.diarise:
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                logger.warning("HF_TOKEN not set; diarisation will be skipped.")
                self._diarise_pipeline = None
            else:
                try:
                    logger.info("Loading diarisation pipeline...")
                    self._diarise_pipeline = whisperx.diarize.DiarizationPipeline(
                        use_auth_token=hf_token, device=self.device
                    )
                except Exception:
                    logger.exception("Failed to load diarisation pipeline; diarisation will be skipped.")
                    self._diarise_pipeline = None
        else:
            self._diarise_pipeline = None

    def unload_model(self) -> None:
        """Release model references and clear GPU cache."""
        logger.info("Unloading transcription models...")
        self._asr_model = None
        self._diarise_pipeline = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _load_diarise_pipeline(self) -> object | None:
        """Return the cached diarisation pipeline, or None if unavailable."""
        return self._diarise_pipeline

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file and return plain transcript text.

        Speaker labels are stripped for simplicity; alignment and optional
        diarisation are still performed so the pipeline is future-proof.
        """
        if self._asr_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        audio = whisperx.load_audio(audio_path)

        logger.debug("Running transcription...")
        result = self._asr_model.transcribe(
            audio,
            batch_size=self.batch_size,
            language=self.language,
        )
        detected_lang = result.get("language", self.language or "unknown")

        logger.debug("Running alignment (language=%s)...", detected_lang)
        align_model, metadata = whisperx.load_align_model(
            language_code=detected_lang, device=self.device
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        del align_model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        diarise_pipeline = self._load_diarise_pipeline()
        if diarise_pipeline is not None:
            logger.debug("Running diarisation...")
            diarise_segments = diarise_pipeline(
                audio,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )
            result = whisperx.assign_word_speakers(diarise_segments, result)

        # Build plain text from segments, stripping speaker labels.
        segments = result.get("segments", [])
        lines: list[str] = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if text:
                lines.append(text)

        return " ".join(lines)

    def transcribe_with_speakers(self, audio_path: str) -> str:
        """Transcribe and return speaker-labelled transcript text.

        Consecutive segments from the same speaker are merged into a single line.
        """
        if self._asr_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        audio = whisperx.load_audio(audio_path)

        result = self._asr_model.transcribe(
            audio,
            batch_size=self.batch_size,
            language=self.language,
        )
        detected_lang = result.get("language", self.language or "unknown")

        align_model, metadata = whisperx.load_align_model(
            language_code=detected_lang, device=self.device
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        del align_model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        diarise_pipeline = self._load_diarise_pipeline()
        if diarise_pipeline is not None:
            diarise_segments = diarise_pipeline(
                audio,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )
            result = whisperx.assign_word_speakers(diarise_segments, result)

        segments = result.get("segments", [])
        txt_lines: list[str] = []
        current_speaker: str | None = None
        current_buf: list[str] = []
        for seg in segments:
            speaker = seg.get("speaker", "SPEAKER_?")
            text = seg.get("text", "").strip()
            if not text:
                continue
            if speaker != current_speaker:
                if current_buf:
                    txt_lines.append(f"{current_speaker}: {' '.join(current_buf)}")
                current_speaker = speaker
                current_buf = [text]
            else:
                current_buf.append(text)
        if current_buf:
            txt_lines.append(f"{current_speaker}: {' '.join(current_buf)}")

        return "\n\n".join(txt_lines)
