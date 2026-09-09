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

For speaker diarization:

```powershell
pip install -r requirements-diarization.txt
```

### Linux

```bash
sudo apt install ffmpeg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For diarization:

```bash
pip install -r requirements-diarization.txt
```

## 2. Basic transcription

```bash
python kt_transcribe.py "recordings/KT Session.mp4"
```

Default model is `large-v3`.

The script automatically chooses:

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

## 4. Speaker identification

`pyannote/speaker-diarization-community-1` requires a free Hugging Face account,
accepting the model's user conditions, and an access token for the initial download.

Set the token as an environment variable rather than putting it in shell history.

PowerShell:

```powershell
$env:HF_TOKEN="hf_..."
python kt_transcribe.py "recording.mp4" --diarize
```

Linux/macOS:

```bash
export HF_TOKEN="hf_..."
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

### No speakers

```bash
python kt_transcribe.py "recording.mp4" \
  --terms technical_terms.txt
```

## Notes

- The original recording is never modified.
- The script does not send the video to an external transcription API.
- Model files are downloaded on first use and cached locally.
- `technical_terms.txt` biases recognition; it does not perform find/replace.
- Keep the JSON output as the canonical/raw machine-readable result. Later AI
  processing should generate documentation from it or the Markdown transcript,
  but should not overwrite it.
