# KT Transcriber - WORK IN PROGRESS - NOT FINISHED

Local transcription for long technical Knowledge Transfer recordings.

## What it produces

For `session.mp4`, the default output folder contains:

- `session.json` — lossless structured transcript, word timestamps, confidence data, speaker assignments and metadata
- `session.md` — readable Markdown transcript
- `session.txt` — plain text
- `session.srt` — subtitles
- `session.vtt` — WebVTT subtitles

The temporary extracted FLAC is deleted unless `--keep-audio` is used.

## 1. Requirements

Python 3.10+ and FFmpeg.

This project currently targets **Windows and macOS**.

### Windows

Install FFmpeg using your preferred package manager, for example:

```powershell
winget install Gyan.FFmpeg
```

Create a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, `requirements.txt` also installs the CUDA 12 cuBLAS and cuDNN 9
runtime DLLs needed by `faster-whisper`/CTranslate2. The script automatically
adds their virtualenv `bin` folders to the process DLL search path. You do not
need to install the full CUDA Toolkit just for this project.

For speaker diarization:

```powershell
pip install -r requirements-diarization.txt
```

### macOS

macOS uses the same script and project files. CUDA/NVIDIA packages are not required.
On Apple Silicon and Intel Macs, `faster-whisper` and pyannote fall back to CPU automatically.

Install FFmpeg and create the virtual environment:

```bash
brew install ffmpeg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For speaker diarization:

```bash
pip install -r requirements-diarization.txt
```

The script intentionally does not force Apple MPS for pyannote. The documented pyannote
GPU path is CUDA, so macOS stays on CPU for predictable compatibility.

## 2. Basic transcription

```bash
python kt_transcribe.py "recordings/KT Session.mp4"
```

Default model is `large-v3`.

For Whisper/CTranslate2, the script automatically chooses:

- NVIDIA CUDA available: `int8_float16`
- otherwise: CPU `int8`

## 3. Technical vocabulary

```bash
python kt_transcribe.py "recordings/KT-lambda.mp4" \
  --terms technical_terms.txt
```

You can combine files and direct terms:

```bash
python kt_transcribe.py "recording.mp4" \
  --terms technical_terms.txt \
  --term "appconfig" \
  --term "document-processing-service"
```

## 4. Hugging Face token and speaker identification

`pyannote/speaker-diarization-community-1` requires a free Hugging Face account,
accepting the model's user conditions, and an access token for the initial model download.

The token is read automatically from a local `.env` file next to `kt_transcribe.py`.
There is intentionally no CLI option for passing the token.

Once configured, speaker diarization needs no additional token arguments:

```bash
python kt_transcribe.py "recording.mp4" --diarize
```

If you know the speaker count:

```bash
python kt_transcribe.py "recording.mp4" --diarize --num-speakers 3
```

Or provide a range:

```bash
python kt_transcribe.py "recording.mp4" \
  --diarize \
  --min-speakers 2 \
  --max-speakers 5
```


### Optional NVIDIA GPU acceleration for pyannote

Speaker diarization has its own device selection and is independent from Whisper:

- `--diarization-device auto` (default) — CUDA when PyTorch can access an NVIDIA GPU, otherwise CPU
- `--diarization-device cuda` — require CUDA and fail clearly if unavailable
- `--diarization-device cpu` — always use CPU

The normal `requirements-diarization.txt` remains cross-platform and is the correct setup
for macOS and CPU-only machines.

On a **Windows machine with an NVIDIA GPU**, replace the CPU PyTorch wheel with
the CUDA wheel after installing the normal diarization requirements:

```powershell
python -m pip uninstall -y torch
python -m pip install -U -r requirements-diarization-nvidia.txt
```

Verify CUDA is visible to PyTorch:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Then run normally:

```powershell
python kt_transcribe.py "recording.mp4" --diarize
```

With `auto`, the log will state whether pyannote is using `cuda` or `cpu`.

## 5. Useful presets

### Accuracy-focused technical KT

```bash
python kt_transcribe.py "recording.mp4" \
  --model large-v3 \
  --terms technical_terms.txt \
  --diarize \
  --beam-size 5 \
  --batch-size 8
```

### GPU memory is tight

```bash
python kt_transcribe.py "recording.mp4" \
  --model large-v3 \
  --compute-type int8_float16 \
  --batch-size 4
```

If that still runs out of VRAM:

```bash
python kt_transcribe.py "recording.mp4" \
  --model large-v3 \
  --compute-type int8_float16 \
  --batch-size 0
```

### Faster English transcription

```bash
python kt_transcribe.py "recording.mp4" \
  --model distil-large-v3
```

Transcription language is permanently fixed to English (`en`) in the script. There is no language CLI option.

### No speakers

```bash
python kt_transcribe.py "recording.mp4" \
  --terms technical_terms.txt
```

## Notes

- `.env` is loaded automatically with `python-dotenv` from the directory containing `kt_transcribe.py`.
- Pyannote acceleration is auto-detected independently: NVIDIA CUDA on supported Windows setups, CPU on macOS.
- The original recording is never modified.
- The script does not send the video to an external transcription API.
- Model files are downloaded on first use and cached locally.
- `technical_terms.txt` biases recognition; it does not perform find/replace.
- Keep the JSON output as the canonical/raw machine-readable result. Later AI
  processing should generate documentation from it or the Markdown transcript,
  but should not overwrite it.
