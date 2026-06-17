# Remove Noise — Video Speaker Isolation

Clean noisy video recordings by removing background sounds (birds, wind, chatter, machinery) and keeping **one target speaker** — without cutting out parts of their voice.

## What it does

This tool processes a video file through a 7-step AI pipeline:

| Step | Tool | Purpose |
|------|------|---------|
| 1 | ffmpeg | Extract audio from video |
| 2 | [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | Remove ambient noise (birds, wind, etc.) |
| 3 | [Demucs](https://github.com/facebookresearch/demucs) | Isolate human vocals from other sounds |
| 4 | [Pyannote](https://github.com/pyannote/pyannote-audio) | Identify different speakers |
| 5 | Custom masking | Cluster-based speaker mask with gap filling |
| 6 | DeepFilterNet | Polish the final speech |
| 7 | ffmpeg | Merge clean audio back into the original video |

The video stream is copied as-is — only the audio is replaced.

## Scripts

| | `process_audio_v1_1.py` | `process_audio_v2.py` |
|---|---|---|
| Method | Demucs vocals + cluster mask | **SepFormer neural separation** |
| Best for | Noisy outdoor recordings | **Two people talking over each other** |
| HF token | Required | Required (speaker matching only) |
| Output | `<name>_final.mp4` | `<name>_v2_final.mp4` |

`process_audio_v1_1.py` is the recommended V1 entry point. `process_audio.py` is a newer variant with continuous similarity masking and extra tuning flags.

Try **V2** if V1 still leaves background conversation audible.

## Requirements

- **Python 3.10+**
- **ffmpeg** — must be installed and available on your `PATH`

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

- **Hugging Face token** — needed for the Pyannote speaker model (free)

## Setup

### 1. Clone / open the project

```bash
cd remove_noise
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 3. Configure your Hugging Face token

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens) and create a token
3. Accept the model terms at:
   - [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)

Copy the example env file and add your token:

```bash
cp .env.example .env
# Edit .env and replace hf_your_token_here with your real token
```

Or export it directly:

```bash
export HF_TOKEN="hf_your_token_here"
```

The script loads `.env` automatically via `python-dotenv`. Never commit `.env` to git.

## Usage

### Interactive (prompts for file path)

```bash
source .venv/bin/activate
python process_audio_v1_1.py
```

You will be asked:

```
Enter video file path: REC-20260616132649.mp4
```

Output is saved as `<filename>_final.mp4` in the same folder.

### Command line (V1)

```bash
python process_audio_v1_1.py input.mp4 -o final.mp4
```

### Lock one speaker from the first 10 seconds (recommended)

Make sure **only the person you want to keep** is talking in the first 10 seconds:

```bash
python process_audio_v1_1.py input.mp4 -o final.mp4 --speaker-mode lock --reference-sec 10
```

This is the default behavior.

### V2 — SepFormer (recommended for 2-speaker overlap)

```bash
python process_audio_v2.py test.mp4
```

Output: `test_v2_final.mp4`

No Demucs or masking. Uses SepFormer to split 2 voices, then keeps the one that matches the first 10 seconds.

```bash
python process_audio_v2.py test.mp4 -o test_v2_final.mp4 --reference-sec 10
```

### Save intermediate files (for debugging)

```bash
python process_audio_v1_1.py input.mp4 -o final.mp4 --keep-intermediates
```

This saves the WAV files at each pipeline stage next to your input video.

### Example

```bash
export HF_TOKEN="hf_your_token_here"
python process_audio_v1_1.py REC-20260616131556.mp4 -o final.mp4
```

Output:

```
[1/7] Extracting audio...
[2/7] DeepFilterNet pass 1 (ambient noise)...
[3/7] Demucs vocal isolation...
[4/7] Loading Pyannote speaker model...
[5/7] Isolating target speaker...
Locked target speaker from first 10.0s using similarity: {0: 0.91, 1: 0.42}
Speaker cluster energy: {0: 21.68, 1: 2.34}
Target speaker cluster: 0
[6/7] DeepFilterNet pass 2 (speech polish)...
[7/7] Merging with video...

Done: final.mp4
```

## How speaker selection works

### `lock` mode (default)

Best when you want **one specific person** and can make sure they speak alone at the start.

1. The cleaned vocal track is split into short windows.
2. Each window gets a **speaker embedding** from Pyannote's WeSpeaker model.
3. Windows are grouped into speaker clusters with agglomerative clustering.
4. The script learns a **reference voice** from speech in the first `10` seconds.
5. The cluster whose centroid best matches that reference is kept.
6. A gap-filled mask is built from that cluster's windows and applied to the vocals.
7. The isolated track is peak-normalized before the final denoise pass.

### `dominant` mode

Keeps the **loudest speaker overall**, even if another person talks more later:

```bash
python process_audio_v1_1.py input.mp4 --speaker-mode dominant
```

## Tuning parameters

If you still hear silence gaps or background voices, adjust these flags:

| Parameter | Default | What it does | If voice is cut / silent | If other voices remain |
|-----------|---------|--------------|--------------------------|------------------------|
| `--reference-sec` | `10` | Seconds used to learn target speaker | Make sure only target speaker talks in this window | Increase if first part is noisy |
| `--min-rms` | `0.004` | Ignore very quiet audio | Lower to `0.003` | Raise to `0.005` |
| `--segment-pad-sec` | `0.7` | Pad before/after kept speech | Raise to `0.9` | Lower to `0.5` |
| `--gap-fill-sec` | `1.2` | Fill short silence inside speech | Raise to `1.4` | Lower to `1.0` |
| `--edge-pad-sec` | `0.2` | Expand mask edges | Raise to `0.25` | Lower to `0.15` |
| `--max-speakers` | `2` | Number of speaker clusters | Keep at `2` for 2-person audio | Keep at `2` |
| `--window-sec` | `1.5` | Analysis window size | Try `1.2` | Try `1.8` |
| `--hop-sec` | `0.2` | Analysis step size | Try `0.15` | Try `0.25` |
| `--fade-sec` | `0.06` | Fade duration at mask boundaries | Raise to `0.08` | Lower to `0.04` |

### Recommended preset for silence gaps

```bash
python process_audio_v1_1.py input.mp4 \
  --speaker-mode lock \
  --reference-sec 10 \
  --min-rms 0.003 \
  --segment-pad-sec 0.9 \
  --gap-fill-sec 1.4 \
  --edge-pad-sec 0.25
```

### Recommended preset for stronger background-voice removal

```bash
python process_audio_v1_1.py input.mp4 \
  --speaker-mode lock \
  --reference-sec 10 \
  --min-rms 0.005 \
  --segment-pad-sec 0.5 \
  --gap-fill-sec 1.0 \
  --max-speakers 2
```

For stricter rejection of non-target voices, try `process_audio.py`, which adds continuous similarity masking, hard muting, and gain controls.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ffmpeg is not installed` | Install ffmpeg (see Requirements) |
| `HF_TOKEN is required` | Set `export HF_TOKEN=...` or pass `--hf-token` |
| Pyannote 403 / download error | Accept model terms on Hugging Face (link above) |
| Voice still has silence gaps | Lower `--min-rms`, raise `--segment-pad-sec` and `--gap-fill-sec` |
| Wrong person kept | Use `--speaker-mode lock` and make only that person talk in first 10s |
| Background voices still audible | Raise `--min-rms`, lower `--segment-pad-sec`, use cleaner reference audio |
| Processing is slow | Normal on CPU; ~2–3× realtime. GPU speeds up Demucs significantly. |
| SSL error downloading Demucs model | Download manually: `curl -L -o ~/.cache/torch/hub/checkpoints/955717e8-8726e21a.th https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th` |

## Project structure

```
remove_noise/
├── process_audio_v1_1.py # V1: Demucs + cluster mask pipeline (recommended)
├── process_audio_v1_0.py # Earlier V1 baseline
├── process_audio.py      # V1 advanced: similarity mask + extra tuning
├── process_audio_v2.py   # V2: SepFormer separation pipeline
├── requirements.txt      # Python dependencies
├── .env.example          # HF token template
├── README.md             # This file
└── .gitignore
```

## Security

- **Never commit your HF token** to git or share it publicly.
- Revoke and recreate your token if it has been exposed.
- Add `.env` to `.gitignore` (already included).

## License

This project uses open-source models with their own licenses:

- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) — MIT / Apache
- [Demucs](https://github.com/facebookresearch/demucs) — MIT
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio) — MIT
