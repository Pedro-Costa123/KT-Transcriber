# KT Transcriber

> **Status:** Testing — native transcription successfully tested on one real KT recording. Docker packaging added and ready for Windows/NVIDIA and macOS testing.

Local transcription for long technical Knowledge Transfer recordings.

## What it produces

For `session.mp4`, the default output folder contains:

- `session.json` — lossless structured transcript, word timestamps, confidence data, speaker assignments and metadata
- `session.md` — readable Markdown transcript
- `session.txt` — plain text
- `session.srt` — subtitles
- `session.vtt` — WebVTT subtitles

The temporary extracted FLAC is deleted unless `--keep-audio` is used.

## Recommended setup: Docker

> **Updating an existing checkout:** keep your current `.env` and customized
> `technical_terms.txt`. The Docker setup uses those same files, so there is no
> need to replace them when adding the Docker files.

Docker is the simplest way to run KT Transcriber on another machine because Python,
FFmpeg, faster-whisper, pyannote, and the relevant runtime dependencies are packaged
inside the image.

The project provides two Docker targets while keeping the same Python script:

- **Windows + NVIDIA GPU:** `kt-transcriber:cuda`
- **macOS:** `kt-transcriber:mac` (CPU)

The original recordings are **not copied into the Docker image**. The local
`recordings/` folder is mounted read-only and generated files are written to
`transcripts/`.

Hugging Face model downloads are stored in a persistent Docker volume, so models do
not need to be downloaded again every time a container is recreated.

### Docker prerequisites

1. Install Docker Desktop.
2. Copy `.env.example` to `.env` and add the Hugging Face token.
3. Put recordings inside the project's `recordings/` folder.
4. Run the Docker command from the project root.

The token file remains ignored by Git.

### Windows + NVIDIA

Docker Desktop must use the WSL2 backend and the NVIDIA driver must support GPU access
from WSL2 containers.

Update WSL first:

```powershell
wsl --update
```

Verify that Docker can see the NVIDIA GPU:

```powershell
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

Copy the environment template if this is the first run:

```powershell
Copy-Item .env.example .env
```

Put the recording in `recordings/`, for example:

```text
recordings/KT Session.mp4
```

Build and run the Windows/NVIDIA container:

```powershell
docker compose --profile windows run --rm --build transcriber-windows `
  "/app/recordings/KT Session.mp4" `
  --output-dir "/app/transcripts/KT Session" `
  --terms /app/technical_terms.txt `
  --diarize
```

Generated files will be available on the host at:

```text
transcripts/KT Session/
```

The Windows Docker image uses:

- CUDA 12 + cuDNN runtime for faster-whisper/CTranslate2
- PyTorch CUDA 13.0 for pyannote, including RTX 50-series / `sm_120` support
- automatic CUDA architecture validation from the Python script

To pass additional KT Transcriber options, append them to the command. For example:

```powershell
docker compose --profile windows run --rm --build transcriber-windows `
  "/app/recordings/KT Session.mp4" `
  --output-dir "/app/transcripts/KT Session" `
  --terms /app/technical_terms.txt `
  --diarize `
  --num-speakers 3
```

### macOS

The macOS Docker image runs the same transcription pipeline on CPU. The image builds
for the Mac's native Docker architecture, including Apple Silicon, without NVIDIA or
CUDA dependencies.

Copy the environment template if this is the first run:

```bash
cp .env.example .env
```

Put the recording in `recordings/`, for example:

```text
recordings/KT Session.mp4
```

Build and run the macOS container:

```bash
docker compose --profile mac run --rm --build transcriber-mac \
  "/app/recordings/KT Session.mp4" \
  --output-dir "/app/transcripts/KT Session" \
  --terms /app/technical_terms.txt \
  --diarize
```

Generated files will be available on the host at:

```text
transcripts/KT Session/
```

> macOS Docker execution currently uses CPU. Docker Desktop does not expose Apple
> Metal/MPS to this PyTorch/CTranslate2 pipeline in the same way the Windows WSL2
> backend exposes NVIDIA CUDA.

### Docker defaults and options

The Docker images use the same defaults as the native Python script:

```text
large-v3
English transcription
VAD
word timestamps
automatic device selection
```

The Docker commands above additionally enable:

```text
technical_terms.txt hotwords
speaker diarization
```

The first run takes longer because Docker must build the image and download the AI
models. Subsequent runs reuse Docker build layers and the persistent Hugging Face model
cache.

### Docker files

```text
docker/
├── Dockerfile.cuda
└── Dockerfile.mac

docker-compose.yml
.dockerignore
recordings/
transcripts/
```

There are intentionally no launcher/wrapper scripts. Run the `docker compose` commands
directly so the Docker configuration remains visible and easy to troubleshoot.

The native Python setup below remains available for development, troubleshooting, or
machines where Docker Desktop cannot be used.

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

Technical vocabulary is passed to Whisper only through `hotwords`. The script intentionally does not use `initial_prompt`, because prompt text can leak into long-form transcripts.

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

The NVIDIA requirements use the PyTorch **CUDA 13.0 (`cu130`)** build. This is
important for RTX 50-series / Blackwell GPUs such as the RTX 5060, whose
compute capability is `sm_120`. Older `cu126` PyTorch wheels can detect these
GPUs while still lacking executable kernels for `sm_120`.

The CUDA version used here is for **PyTorch/pyannote only**. `faster-whisper`
uses CTranslate2 and its own CUDA 12 runtime packages, so the two runtimes can
coexist in the same virtual environment.

Verify CUDA is visible **and that the wheel supports the GPU architecture**:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('Capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'none'); print('Architectures:', torch.cuda.get_arch_list() if torch.cuda.is_available() else [])"
```

For an RTX 5060, the capability should be `(12, 0)` and the architecture list
must include `sm_120`.

Then run normally:

```powershell
python kt_transcribe.py "recording.mp4" --diarize
```

With `auto`, the script checks both CUDA availability and whether the installed
PyTorch wheel contains kernels for the detected GPU architecture. If CUDA can
see the GPU but the wheel does not support its architecture, diarization safely
falls back to CPU instead of failing later during a CUDA kernel launch.

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

- Whisper uses technical terms only as `hotwords`; no generated `initial_prompt` is sent to the model.
- `.env` is loaded automatically with `python-dotenv` from the directory containing `kt_transcribe.py`.
- Pyannote acceleration is auto-detected independently: NVIDIA CUDA on supported Windows setups, CPU on macOS.
- The original recording is never modified.
- The script does not send the video to an external transcription API.
- Model files are downloaded on first use and cached locally.
- `technical_terms.txt` biases recognition; it does not perform find/replace.
- Keep the JSON output as the canonical/raw machine-readable result. Later AI
  processing should generate documentation from it or the Markdown transcript,
  but should not overwrite it.
