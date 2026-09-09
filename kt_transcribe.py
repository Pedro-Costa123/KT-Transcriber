#!/usr/bin/env python3
"""
kt_transcribe.py

Local-first long-form transcription for technical Knowledge Transfer recordings.

Pipeline:
  media file -> FFmpeg 16 kHz mono FLAC -> faster-whisper -> optional pyannote
  speaker diarization -> JSON / Markdown / TXT / SRT / VTT

Designed for large recordings (multi-hour MP4/MKV/MOV files) without uploading
the original video to any external transcription service.

Python: 3.10+
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from dotenv import load_dotenv


LOG = logging.getLogger("kt-transcriber")

# Keep AddDllDirectory handles alive for the lifetime of the process on Windows.
_WINDOWS_DLL_DIR_HANDLES: list[Any] = []


def configure_windows_cuda_dll_search() -> list[Path]:
    """
    Add CUDA runtime directories installed by NVIDIA pip wheels to Windows'
    DLL search path before CTranslate2/pyannote/PyTorch load native libraries.

    This makes a project-local virtualenv sufficient for faster-whisper GPU
    execution; a system-wide CUDA Toolkit installation is not required.
    """
    if os.name != "nt":
        return []

    candidates = [
        Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
        Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
        Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
    ]
    found = [path for path in candidates if path.is_dir()]
    if not found:
        return []

    # Native dependencies loaded later by CTranslate2 use the process DLL
    # search path. Prepend the wheel directories to PATH and also register
    # them with Python's Windows DLL directory API.
    current_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(str(p) for p in found) + os.pathsep + current_path

    if hasattr(os, "add_dll_directory"):
        for directory in found:
            try:
                _WINDOWS_DLL_DIR_HANDLES.append(os.add_dll_directory(str(directory)))
            except OSError:
                pass

    return found


def find_windows_dll(name: str) -> Optional[Path]:
    if os.name != "nt":
        return None

    for raw_dir in os.environ.get("PATH", "").split(os.pathsep):
        if not raw_dir:
            continue
        try:
            candidate = Path(raw_dir.strip('"')) / name
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def validate_windows_cuda_runtime(device: str) -> None:
    """Fail early with an actionable message when CUDA 12 DLLs are absent."""
    if os.name != "nt" or device != "cuda":
        return

    required = ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll")
    missing = [name for name in required if find_windows_dll(name) is None]
    if not missing:
        return

    raise RuntimeError(
        "CUDA GPU mode was selected, but required CUDA 12/cuDNN 9 DLLs "
        f"are missing: {', '.join(missing)}\n\n"
        "Activate this project's virtual environment and run:\n"
        "  python -m pip install -U nvidia-cublas-cu12 nvidia-cudnn-cu12\n\n"
        "Then close/reopen the terminal (or reactivate the venv) and run the "
        "transcriber again. The script automatically adds the installed "
        "nvidia\\cublas\\bin and nvidia\\cudnn\\bin directories to "
        "the Windows DLL search path.\n\n"
        "To run without NVIDIA GPU acceleration instead, use: --device cpu"
    )


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_hms(seconds: float, millis: bool = False) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    ms = int(round((seconds - whole) * 1000))

    if ms == 1000:
        whole += 1
        ms = 0

    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)

    if millis:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_srt_time(seconds: float) -> str:
    return format_hms(seconds, millis=True).replace(".", ",")


def format_vtt_time(seconds: float) -> str:
    return format_hms(seconds, millis=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg was not found in PATH.\n"
            "Install FFmpeg first, then re-run this command."
        )


def get_media_duration(path: Path) -> Optional[float]:
    """Best-effort duration using ffprobe if available."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None

    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def extract_audio(source: Path, audio_path: Path, overwrite: bool) -> None:
    """
    Convert arbitrary media to mono 16 kHz FLAC.

    16 kHz mono is enough for Whisper/pyannote speech processing and avoids
    carrying multi-GB video through the ML pipeline.
    """
    check_ffmpeg()
    ensure_parent(audio_path)

    if audio_path.exists() and not overwrite:
        LOG.info("Reusing existing extracted audio: %s", audio_path)
        return

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(source),
        "-vn",
        "-map_metadata", "-1",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "flac",
        "-compression_level", "8",
        str(audio_path),
    ]

    LOG.info("Extracting 16 kHz mono FLAC...")
    subprocess.run(cmd, check=True)


def load_terms(files: list[Path], inline_terms: list[str]) -> list[str]:
    """
    Load technical terms from files and --term arguments.

    File format:
      - one term per line
      - blank lines ignored
      - lines beginning with # ignored
      - comma-separated values on one line are also accepted
    """
    terms: list[str] = []

    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"Technical terms file not found: {path}")

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            for value in line.split(","):
                value = value.strip()
                if value:
                    terms.append(value)

    terms.extend(t.strip() for t in inline_terms if t.strip())

    # De-duplicate while preserving order/casing of the first occurrence.
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(term)

    return unique


def build_hotwords(terms: list[str], max_chars: int = 2500) -> Optional[str]:
    """
    faster-whisper accepts hotwords as a hint string. Keep it bounded so that a
    giant vocabulary file cannot consume the model's prompt budget.
    """
    if not terms:
        return None

    selected: list[str] = []
    current = 0

    for term in terms:
        extra = len(term) + (2 if selected else 0)
        if current + extra > max_chars:
            break
        selected.append(term)
        current += extra

    return ", ".join(selected) if selected else None


def build_initial_prompt(
    user_prompt: Optional[str],
    terms: list[str],
    max_terms: int = 80,
) -> Optional[str]:
    parts: list[str] = []

    if user_prompt:
        parts.append(user_prompt.strip())

    if terms:
        short_terms = ", ".join(terms[:max_terms])
        parts.append(
            "This is a technical software-engineering knowledge-transfer session. "
            "Preserve product names, service names, acronyms, API names, database "
            f"names, cloud terminology, identifiers, and technical vocabulary. "
            f"Known terms include: {short_terms}."
        )

    return " ".join(p for p in parts if p) or None


# ---------------------------------------------------------------------------
# Device / model configuration
# ---------------------------------------------------------------------------

def cuda_available_for_ctranslate2() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if cuda_available_for_ctranslate2() else "cpu"


def resolve_compute_type(device: str, requested: str) -> str:
    if requested != "auto":
        return requested

    # A good default for large Whisper models on 8 GB-class NVIDIA GPUs.
    if device == "cuda":
        return "int8_float16"
    return "int8"


# ---------------------------------------------------------------------------
# faster-whisper transcription
# ---------------------------------------------------------------------------

def serialize_word(word: Any) -> dict[str, Any]:
    return {
        "start": float(word.start),
        "end": float(word.end),
        "word": word.word,
        "probability": (
            float(word.probability)
            if getattr(word, "probability", None) is not None
            else None
        ),
    }


def serialize_segment(segment: Any) -> dict[str, Any]:
    words = getattr(segment, "words", None)
    return {
        "id": int(segment.id),
        "start": float(segment.start),
        "end": float(segment.end),
        "text": segment.text.strip(),
        "avg_logprob": (
            float(segment.avg_logprob)
            if getattr(segment, "avg_logprob", None) is not None
            else None
        ),
        "compression_ratio": (
            float(segment.compression_ratio)
            if getattr(segment, "compression_ratio", None) is not None
            else None
        ),
        "no_speech_prob": (
            float(segment.no_speech_prob)
            if getattr(segment, "no_speech_prob", None) is not None
            else None
        ),
        "words": [serialize_word(w) for w in words] if words else [],
    }


def perform_transcription(
    audio_path: Path,
    *,
    model_name: str,
    device: str,
    compute_type: str,
    task: str,
    beam_size: int,
    batch_size: int,
    word_timestamps: bool,
    vad: bool,
    vad_min_silence_ms: int,
    condition_on_previous_text: bool,
    hotwords: Optional[str],
    initial_prompt: Optional[str],
    cpu_threads: int,
    download_root: Optional[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from faster_whisper import BatchedInferencePipeline, WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. "
            "Run: pip install -r requirements.txt"
        ) from exc

    LOG.info(
        "Loading Whisper model '%s' on %s (%s)...",
        model_name, device, compute_type
    )

    model_kwargs: dict[str, Any] = {
        "device": device,
        "compute_type": compute_type,
        "cpu_threads": cpu_threads,
    }
    if download_root is not None:
        model_kwargs["download_root"] = str(download_root)

    model = WhisperModel(model_name, **model_kwargs)

    transcribe_kwargs: dict[str, Any] = {
        "language": "en",
        "task": task,
        "beam_size": beam_size,
        "word_timestamps": word_timestamps,
        "vad_filter": vad,
        "condition_on_previous_text": condition_on_previous_text,
        "hotwords": hotwords,
        "initial_prompt": initial_prompt,
    }

    if vad:
        transcribe_kwargs["vad_parameters"] = {
            "min_silence_duration_ms": vad_min_silence_ms
        }

    if batch_size > 0:
        LOG.info("Using batched inference (batch_size=%d).", batch_size)
        transcriber = BatchedInferencePipeline(model=model)
        segments_gen, info = transcriber.transcribe(
            str(audio_path),
            batch_size=batch_size,
            **transcribe_kwargs,
        )
    else:
        LOG.info("Using standard (non-batched) inference.")
        segments_gen, info = model.transcribe(
            str(audio_path),
            **transcribe_kwargs,
        )

    # Iterating the generator performs the actual transcription.
    segments: list[dict[str, Any]] = []
    last_report = -1
    duration = float(getattr(info, "duration", 0.0) or 0.0)

    LOG.info("Transcribing...")
    for segment in segments_gen:
        item = serialize_segment(segment)
        segments.append(item)

        if duration > 0:
            pct = min(100, int((item["end"] / duration) * 100))
            bucket = pct // 5
            if bucket > last_report:
                last_report = bucket
                LOG.info(
                    "Progress: %3d%%  (%s / %s)",
                    pct,
                    format_hms(item["end"]),
                    format_hms(duration),
                )

    metadata = {
        "language": getattr(info, "language", "en"),
        "language_probability": (
            float(getattr(info, "language_probability", 0.0))
            if getattr(info, "language_probability", None) is not None
            else None
        ),
        "duration": duration,
        "duration_after_vad": (
            float(getattr(info, "duration_after_vad", 0.0))
            if getattr(info, "duration_after_vad", None) is not None
            else None
        ),
    }

    # Explicitly let CT2/model memory go before optional pyannote loads.
    del model
    if "transcriber" in locals():
        del transcriber
    gc.collect()

    return segments, metadata


# ---------------------------------------------------------------------------
# pyannote speaker diarization
# ---------------------------------------------------------------------------

def diarize_audio(
    audio_path: Path,
    *,
    model_name: str,
    hf_token: Optional[str],
    diarization_device: str,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> list[dict[str, Any]]:
    try:
        import soundfile as sf
        import torch
        from pyannote.audio import Pipeline
        from pyannote.audio.pipelines.utils.hook import ProgressHook
    except ImportError as exc:
        raise RuntimeError(
            "Speaker diarization dependencies are not installed. "
            "Run: pip install -r requirements-diarization.txt"
        ) from exc

    if not hf_token and not Path(model_name).exists():
        raise RuntimeError(
            "Speaker diarization needs a Hugging Face token the first time the "
            "Community-1 model is downloaded.\n"
            "Set HF_TOKEN in the .env file next to kt_transcribe.py.\n"
            "Also accept the model's user conditions on Hugging Face first."
        )

    LOG.info("Loading diarization model '%s'...", model_name)
    pipeline = Pipeline.from_pretrained(model_name, token=hf_token)

    # Diarization acceleration is independent from faster-whisper/CTranslate2.
    # This keeps the script portable: macOS uses CPU, while Windows machines with a CUDA-enabled
    # PyTorch build can use NVIDIA when the wheel supports the GPU architecture.
    torch_cuda_available = bool(torch.cuda.is_available())
    cuda_arch_supported = False
    gpu_name = None
    gpu_cc = None
    torch_arches: list[str] = []

    if torch_cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            gpu_cc = f"sm_{major}{minor}"
            torch_arches = list(torch.cuda.get_arch_list())
            cuda_arch_supported = gpu_cc in torch_arches
        except Exception as exc:
            LOG.warning("Could not validate PyTorch CUDA architecture support: %s", exc)

    requested = diarization_device.lower()
    if requested == "auto":
        if torch_cuda_available and cuda_arch_supported:
            diar_device = "cuda"
        else:
            diar_device = "cpu"
            if torch_cuda_available and gpu_cc and not cuda_arch_supported:
                LOG.warning(
                    "PyTorch can see GPU '%s' (%s), but this PyTorch CUDA wheel "
                    "does not include kernels for that architecture. "
                    "Falling back to CPU for diarization.",
                    gpu_name or "unknown",
                    gpu_cc,
                )
    elif requested == "cuda":
        if not torch_cuda_available:
            raise RuntimeError(
                "--diarization-device cuda was requested, but this PyTorch build "
                "cannot access CUDA. Install a CUDA-enabled PyTorch build on a "
                "Windows NVIDIA machine, or use --diarization-device cpu."
            )
        if not cuda_arch_supported:
            raise RuntimeError(
                "--diarization-device cuda was requested, but the installed "
                f"PyTorch build does not support this GPU architecture ({gpu_cc or 'unknown'}).\n"
                f"GPU: {gpu_name or 'unknown'}\n"
                f"PyTorch: {torch.__version__}\n"
                f"PyTorch CUDA build: {getattr(torch.version, 'cuda', None) or 'none'}\n"
                f"Architectures in this wheel: {', '.join(torch_arches) or 'unknown'}\n"
                "For RTX 50-series / Blackwell GPUs on Windows, install the "
                "project's CUDA 13 PyTorch requirements:\n"
                "  python -m pip uninstall -y torch\n"
                "  python -m pip install -U -r requirements-diarization-nvidia.txt"
            )
        diar_device = "cuda"
    else:
        diar_device = "cpu"

    LOG.info(
        "PyTorch: %s | platform: %s %s | CUDA build: %s | CUDA available: %s",
        torch.__version__,
        platform.system(),
        platform.machine(),
        getattr(torch.version, "cuda", None) or "none",
        torch_cuda_available,
    )

    if torch_cuda_available:
        LOG.info(
            "PyTorch CUDA GPU: %s | capability: %s | architecture supported: %s",
            gpu_name or "unknown",
            gpu_cc or "unknown",
            cuda_arch_supported,
        )
        if torch_arches:
            LOG.debug("PyTorch wheel CUDA architectures: %s", ", ".join(torch_arches))

    # Apple Silicon may expose PyTorch MPS, but pyannote's documented acceleration
    # path is CUDA. Keep macOS on CPU for predictable cross-platform behavior.
    if platform.system() == "Darwin":
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                LOG.info(
                    "Apple MPS is available, but this script uses CPU for pyannote "
                    "on macOS for compatibility."
                )
        except Exception:
            pass

    if diar_device == "cuda":
        pipeline.to(torch.device("cuda"))
    LOG.info("Running speaker diarization on %s...", diar_device)

    kwargs: dict[str, Any] = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    # pyannote.audio 4 uses TorchCodec when it receives a file path.
    # TorchCodec requires a shared-library FFmpeg build on Windows and can be
    # fragile across PyTorch/FFmpeg combinations. We already have a clean mono
    # 16 kHz FLAC from the extraction stage, so load it ourselves and give
    # pyannote the officially supported in-memory waveform mapping instead.
    # This bypasses TorchCodec decoding completely and works cross-platform.
    LOG.info("Loading extracted audio into memory for diarization...")
    audio_samples, sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=True,
    )

    # soundfile returns (time, channels); pyannote expects (channels, time).
    waveform = torch.from_numpy(audio_samples.T)
    audio_input = {
        "waveform": waveform,
        "sample_rate": int(sample_rate),
        "uri": audio_path.stem,
    }

    duration_seconds = waveform.shape[1] / float(sample_rate)
    memory_mib = waveform.numel() * waveform.element_size() / (1024 * 1024)
    LOG.info(
        "Diarization audio loaded: %s, %.1f MiB in memory.",
        format_hms(duration_seconds),
        memory_mib,
    )

    with ProgressHook() as hook:
        output = pipeline(audio_input, hook=hook, **kwargs)

    # Pipeline output no longer needs the source waveform. Release it before
    # building the final transcript structures.
    del audio_input, waveform, audio_samples

    # Community-1 provides an exclusive diarization that is easier to reconcile
    # with transcription timestamps. Fall back to normal diarization if needed.
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = output.speaker_diarization

    turns: list[dict[str, Any]] = []
    for turn, speaker in annotation:
        turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        })

    LOG.info(
        "Diarization complete: %d turns, %d detected speakers.",
        len(turns),
        len({t["speaker"] for t in turns}),
    )

    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return turns


# ---------------------------------------------------------------------------
# Speaker assignment / utterance building
# ---------------------------------------------------------------------------

def overlap_seconds(
    a_start: float, a_end: float, b_start: float, b_end: float
) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def best_speaker(
    start: float,
    end: float,
    turns: list[dict[str, Any]],
) -> Optional[str]:
    if not turns:
        return None

    best: Optional[str] = None
    best_overlap = 0.0

    for turn in turns:
        if turn["end"] < start:
            continue
        if turn["start"] > end:
            break

        ov = overlap_seconds(start, end, turn["start"], turn["end"])
        if ov > best_overlap:
            best_overlap = ov
            best = turn["speaker"]

    # If a tiny word/segment falls between diarization boundaries, use midpoint.
    if best is None:
        midpoint = (start + end) / 2.0
        nearest_distance = float("inf")
        for turn in turns:
            if turn["start"] <= midpoint <= turn["end"]:
                return turn["speaker"]

            distance = min(
                abs(midpoint - turn["start"]),
                abs(midpoint - turn["end"]),
            )
            if distance < nearest_distance:
                nearest_distance = distance
                best = turn["speaker"]

        # Do not assign a very distant speaker across long silent gaps.
        if nearest_distance > 2.0:
            return None

    return best


def assign_speakers(
    segments: list[dict[str, Any]],
    turns: list[dict[str, Any]],
) -> None:
    for segment in segments:
        segment["speaker"] = best_speaker(
            segment["start"], segment["end"], turns
        )

        for word in segment.get("words", []):
            word["speaker"] = best_speaker(
                word["start"], word["end"], turns
            )


def append_text(existing: str, token: str) -> str:
    """
    faster-whisper word tokens usually preserve their leading whitespace.
    Concatenating rather than " ".join() keeps punctuation natural.
    """
    if not existing:
        return token.strip()
    return (existing + token).strip()


def build_utterances(
    segments: list[dict[str, Any]],
    diarization_enabled: bool,
    merge_gap_seconds: float = 1.2,
) -> list[dict[str, Any]]:
    """
    Build speaker-aware chunks.

    If word timestamps exist, speaker changes can split a Whisper segment.
    Otherwise each Whisper segment becomes an utterance.
    """
    atomic: list[dict[str, Any]] = []

    for segment in segments:
        words = segment.get("words", [])
        if words:
            current: Optional[dict[str, Any]] = None

            for word in words:
                speaker = (
                    word.get("speaker")
                    or segment.get("speaker")
                    or ("UNKNOWN" if diarization_enabled else None)
                )

                if (
                    current is None
                    or current["speaker"] != speaker
                    or word["start"] - current["end"] > merge_gap_seconds
                ):
                    if current:
                        atomic.append(current)
                    current = {
                        "start": word["start"],
                        "end": word["end"],
                        "speaker": speaker,
                        "text": word["word"].strip(),
                    }
                else:
                    current["end"] = word["end"]
                    current["text"] = append_text(
                        current["text"], word["word"]
                    )

            if current:
                atomic.append(current)
        else:
            atomic.append({
                "start": segment["start"],
                "end": segment["end"],
                "speaker": (
                    segment.get("speaker")
                    or ("UNKNOWN" if diarization_enabled else None)
                ),
                "text": segment["text"].strip(),
            })

    # Merge adjacent chunks from the same speaker across Whisper segment borders.
    merged: list[dict[str, Any]] = []
    for item in atomic:
        if not item["text"]:
            continue

        if (
            merged
            and merged[-1]["speaker"] == item["speaker"]
            and item["start"] - merged[-1]["end"] <= merge_gap_seconds
        ):
            merged[-1]["end"] = item["end"]
            if item["text"]:
                merged[-1]["text"] = (
                    merged[-1]["text"].rstrip()
                    + " "
                    + item["text"].lstrip()
                ).strip()
        else:
            merged.append(item.copy())

    return merged


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def speaker_prefix(item: dict[str, Any]) -> str:
    speaker = item.get("speaker")
    return f"{speaker}: " if speaker else ""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_markdown(
    path: Path,
    title: str,
    payload: dict[str, Any],
) -> None:
    ensure_parent(path)

    meta = payload["metadata"]
    lines = [
        f"# {title}",
        "",
        "## Recording metadata",
        "",
        f"- Source: `{meta['source_file']}`",
        f"- Model: `{meta['model']}`",
        f"- Device: `{meta['device']}` / `{meta['compute_type']}`",
        f"- Language: `{meta.get('language') or 'auto/unknown'}`",
        f"- Duration: `{format_hms(meta.get('duration') or 0)}`",
        f"- Speaker diarization: `{'yes' if meta['diarization'] else 'no'}`",
        "",
        "## Transcript",
        "",
    ]

    last_speaker = object()
    for item in payload["utterances"]:
        timestamp = format_hms(item["start"])
        speaker = item.get("speaker")

        if speaker and speaker != last_speaker:
            lines.append(f"### {speaker}")
            lines.append("")

        lines.append(f"**[{timestamp}]** {item['text']}")
        lines.append("")
        last_speaker = speaker

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_txt(path: Path, utterances: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    lines = []
    for item in utterances:
        lines.append(
            f"[{format_hms(item['start'])}] "
            f"{speaker_prefix(item)}{item['text']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_srt(path: Path, utterances: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    blocks: list[str] = []
    for i, item in enumerate(utterances, start=1):
        blocks.append(str(i))
        blocks.append(
            f"{format_srt_time(item['start'])} --> "
            f"{format_srt_time(item['end'])}"
        )
        blocks.append(f"{speaker_prefix(item)}{item['text']}")
        blocks.append("")
    path.write_text("\n".join(blocks), encoding="utf-8")


def write_vtt(path: Path, utterances: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    lines = ["WEBVTT", ""]
    for item in utterances:
        lines.append(
            f"{format_vtt_time(item['start'])} --> "
            f"{format_vtt_time(item['end'])}"
        )
        lines.append(f"{speaker_prefix(item)}{item['text']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    out_dir: Path,
    stem: str,
    formats: set[str],
    title: str,
    payload: dict[str, Any],
) -> list[Path]:
    written: list[Path] = []

    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"

        if fmt == "json":
            write_json(path, payload)
        elif fmt == "md":
            write_markdown(path, title, payload)
        elif fmt == "txt":
            write_txt(path, payload["utterances"])
        elif fmt == "srt":
            write_srt(path, payload["utterances"])
        elif fmt == "vtt":
            write_vtt(path, payload["utterances"])
        else:
            raise ValueError(f"Unsupported output format: {fmt}")

        written.append(path)

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_formats(value: str) -> set[str]:
    allowed = {"json", "md", "txt", "srt", "vtt"}
    result = {v.strip().lower() for v in value.split(",") if v.strip()}
    invalid = result - allowed
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported format(s): {', '.join(sorted(invalid))}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe long technical recordings locally with faster-whisper, "
            "with optional pyannote speaker diarization."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input", type=Path, help="Input video/audio file")
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input-folder>/transcripts/<stem>)",
    )
    parser.add_argument("--title", default=None, help="Markdown document title")

    # Whisper
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model name/path",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help=(
            "CTranslate2 compute type. Auto = int8_float16 on CUDA, int8 on CPU"
        ),
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
    )
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Set 0 for non-batched inference",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="0 lets CTranslate2 choose",
    )
    parser.add_argument(
        "--word-timestamps",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Silero voice-activity detection",
    )
    parser.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=500,
        help="Minimum silence duration removed by VAD",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Previous decoded text conditions the next window. Disabled by "
            "default to reduce repetition loops on very long recordings."
        ),
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=None,
        help="Optional model cache directory",
    )

    # Technical vocabulary
    parser.add_argument(
        "--terms",
        type=Path,
        action="append",
        default=[],
        help="Technical terms file; may be specified multiple times",
    )
    parser.add_argument(
        "--term",
        action="append",
        default=[],
        help="Add one technical hotword directly; may be repeated",
    )
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Extra context/hint text for Whisper",
    )

    # Diarization
    parser.add_argument(
        "--diarize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Identify speakers using pyannote Community-1",
    )
    parser.add_argument(
        "--diarization-model",
        default="pyannote/speaker-diarization-community-1",
        help="Hugging Face model name or local model directory",
    )
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)

    # Files / outputs
    parser.add_argument(
        "--formats",
        type=parse_formats,
        default={"json", "md", "txt", "srt", "vtt"},
        help="Comma-separated output formats",
    )
    parser.add_argument(
        "--keep-audio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep extracted FLAC after transcription",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=1.2,
        help="Merge adjacent same-speaker text within this many seconds",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    return parser


def main() -> int:
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=False)

    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    source = args.input.expanduser().resolve()
    if not source.exists():
        parser.error(f"Input file does not exist: {source}")
    if not source.is_file():
        parser.error(f"Input path is not a file: {source}")

    if args.num_speakers is not None and (
        args.min_speakers is not None or args.max_speakers is not None
    ):
        parser.error(
            "--num-speakers cannot be combined with --min-speakers/--max-speakers"
        )

    cuda_dll_dirs = configure_windows_cuda_dll_search()
    if cuda_dll_dirs:
        LOG.info(
            "Configured Windows CUDA DLL directories: %s",
            "; ".join(str(p) for p in cuda_dll_dirs),
        )

    stem = source.stem
    out_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else source.parent / "transcripts" / stem
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_path = out_dir / f"{stem}.16k-mono.flac"
    title = args.title or f"KT Transcript — {stem}"

    device = resolve_device(args.device)
    validate_windows_cuda_runtime(device)
    compute_type = resolve_compute_type(device, args.compute_type)

    terms = load_terms(args.terms, args.term)
    hotwords = build_hotwords(terms)
    initial_prompt = build_initial_prompt(args.initial_prompt, terms)

    if terms:
        LOG.info("Loaded %d technical terms/hotwords.", len(terms))

    LOG.info("Input: %s", source)
    LOG.info("Output: %s", out_dir)
    LOG.info("Device: %s | compute type: %s", device, compute_type)

    source_duration = get_media_duration(source)
    if source_duration:
        LOG.info("Source duration: %s", format_hms(source_duration))

    started = time.perf_counter()

    try:
        extract_audio(source, audio_path, overwrite=args.overwrite)

        segments, transcription_meta = perform_transcription(
            audio_path,
            model_name=args.model,
            device=device,
            compute_type=compute_type,
            task=args.task,
            beam_size=args.beam_size,
            batch_size=args.batch_size,
            word_timestamps=args.word_timestamps,
            vad=args.vad,
            vad_min_silence_ms=args.vad_min_silence_ms,
            condition_on_previous_text=args.condition_on_previous_text,
            hotwords=hotwords,
            initial_prompt=initial_prompt,
            cpu_threads=args.cpu_threads,
            download_root=args.download_root,
        )

        turns: list[dict[str, Any]] = []
        if args.diarize:
            hf_token = os.getenv("HF_TOKEN")
            turns = diarize_audio(
                audio_path,
                model_name=args.diarization_model,
                hf_token=hf_token,
                diarization_device=args.diarization_device,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
            )
            assign_speakers(segments, turns)

        utterances = build_utterances(
            segments,
            diarization_enabled=args.diarize,
            merge_gap_seconds=args.merge_gap,
        )

        elapsed = time.perf_counter() - started

        metadata = {
            "created_at": utc_now_iso(),
            "source_file": str(source),
            "source_size_bytes": source.stat().st_size,
            "source_duration_seconds": source_duration,
            "model": args.model,
            "device": device,
            "compute_type": compute_type,
            "task": args.task,
            "language_requested": "en",
            "language": transcription_meta.get("language"),
            "language_probability": transcription_meta.get("language_probability"),
            "duration": transcription_meta.get("duration") or source_duration,
            "duration_after_vad": transcription_meta.get("duration_after_vad"),
            "beam_size": args.beam_size,
            "batch_size": args.batch_size,
            "vad": args.vad,
            "vad_min_silence_ms": args.vad_min_silence_ms if args.vad else None,
            "condition_on_previous_text": args.condition_on_previous_text,
            "word_timestamps": args.word_timestamps,
            "diarization": args.diarize,
            "diarization_model": args.diarization_model if args.diarize else None,
            "diarization_device_requested": args.diarization_device if args.diarize else None,
            "num_speakers_detected": (
                len({t["speaker"] for t in turns}) if turns else None
            ),
            "technical_terms_count": len(terms),
            "elapsed_seconds": elapsed,
        }

        payload = {
            "metadata": metadata,
            "technical_terms": terms,
            "segments": segments,
            "diarization_turns": turns,
            "utterances": utterances,
        }

        written = write_outputs(
            out_dir=out_dir,
            stem=stem,
            formats=args.formats,
            title=title,
            payload=payload,
        )

        LOG.info("Finished in %s.", format_hms(elapsed))
        for path in sorted(written):
            LOG.info("Created: %s", path)

        if not args.keep_audio:
            try:
                audio_path.unlink(missing_ok=True)
                LOG.info("Removed temporary extracted audio.")
            except OSError as exc:
                LOG.warning("Could not remove temporary audio: %s", exc)

        return 0

    except KeyboardInterrupt:
        LOG.error("Interrupted.")
        return 130
    except subprocess.CalledProcessError as exc:
        LOG.error("FFmpeg failed with exit code %s.", exc.returncode)
        return exc.returncode or 1
    except Exception as exc:
        LOG.exception("Transcription failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
