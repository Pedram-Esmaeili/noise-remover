#!/usr/bin/env python3
"""
Isolate a target speaker in a video and remove background noise.

Pipeline:
  1. Extract audio (ffmpeg)
  2. DeepFilterNet pass 1  — remove birds, wind, ambient noise
  3. Demucs vocal isolation — separate human voice from other sounds
  4. Pyannote speaker ID    — identify speakers and lock target voice
  5. Gap-filled masking     — prevent voice dropouts
  6. DeepFilterNet pass 2   — polish the final speech
  7. Merge clean audio back into the video (ffmpeg)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Inference
from scipy.ndimage import binary_closing, binary_dilation
from sklearn.cluster import AgglomerativeClustering


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def deepfilter(in_path: Path, out_path: Path) -> None:
    from df.enhance import enhance, init_df, load_audio, save_audio

    model, df_state, _ = init_df()
    audio, _ = load_audio(str(in_path), sr=df_state.sr())
    enhanced = enhance(model, df_state, audio)
    save_audio(str(out_path), enhanced, df_state.sr())
    print(f"DeepFilterNet: {in_path.name} -> {out_path.name}", flush=True)


def extract_audio(video: Path, wav: Path, sample_rate: int = 48000) -> None:
    run([
        "ffmpeg", "-y",
        "-i", str(video),
        "-vn", "-ar", str(sample_rate), "-ac", "1",
        "-c:a", "pcm_s16le",
        str(wav),
    ])


def demucs_vocals(wav: Path, out_dir: Path) -> Path:
    run([
        sys.executable, "-m", "demucs",
        "--two-stems=vocals",
        "-n", "htdemucs",
        "-o", str(out_dir),
        str(wav),
    ])
    return out_dir / "htdemucs" / wav.stem / "vocals.wav"


def merge_audio_video(video: Path, audio: Path, output: Path, bitrate: str = "192k") -> None:
    run([
        "ffmpeg", "-y",
        "-i", str(video),
        "-i", str(audio),
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:a", "aac", "-b:a", bitrate,
        "-shortest",
        str(output),
    ])


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def collect_segments(
    audio: np.ndarray,
    sr: int,
    inference: Inference,
    *,
    window_sec: float,
    hop_sec: float,
    min_rms: float,
) -> tuple[list[tuple[int, int, float]], list[np.ndarray]]:
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)
    waveform = torch.from_numpy(audio).unsqueeze(0)

    segments: list[tuple[int, int, float]] = []
    embeddings: list[np.ndarray] = []

    for start in range(0, len(audio) - win + 1, hop):
        chunk = audio[start : start + win]
        energy = rms(chunk)
        if energy < min_rms:
            continue

        excerpt = waveform[:, start : start + win]
        with torch.no_grad():
            emb = inference({"waveform": excerpt, "sample_rate": sr})
        emb = np.asarray(emb).reshape(-1)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        segments.append((start, start + win, energy))
        embeddings.append(emb)

    return segments, embeddings


def cluster_centroids(
    embeddings: list[np.ndarray],
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    for label in np.unique(labels):
        cluster_embs = np.vstack([emb for emb, lbl in zip(embeddings, labels) if lbl == label])
        centroid = cluster_embs.mean(axis=0)
        centroids[int(label)] = centroid / (np.linalg.norm(centroid) + 1e-8)
    return centroids


def reference_embedding(
    segments: list[tuple[int, int, float]],
    embeddings: list[np.ndarray],
    sr: int,
    reference_sec: float,
) -> np.ndarray | None:
    reference_end = int(reference_sec * sr)
    weighted = np.zeros_like(embeddings[0], dtype=np.float64)
    total_energy = 0.0

    for (start, _, energy), emb in zip(segments, embeddings):
        if start >= reference_end:
            continue
        weighted += emb * energy
        total_energy += energy

    if total_energy <= 0:
        return None

    ref = weighted / total_energy
    return ref / (np.linalg.norm(ref) + 1e-8)


def choose_target_cluster(
    segments: list[tuple[int, int, float]],
    embeddings: list[np.ndarray],
    labels: np.ndarray,
    *,
    speaker_mode: str,
    reference_sec: float,
    sr: int,
) -> tuple[int, dict[int, float]]:
    cluster_energy: dict[int, float] = {}
    for (_, _, energy), label in zip(segments, labels):
        cluster_energy[int(label)] = cluster_energy.get(int(label), 0.0) + energy

    centroids = cluster_centroids(embeddings, labels)

    if speaker_mode == "lock":
        ref = reference_embedding(segments, embeddings, sr, reference_sec)
        if ref is not None:
            scores = {
                label: cosine_similarity(centroid, ref)
                for label, centroid in centroids.items()
            }
            target = max(scores, key=scores.get)
            print(
                f"Locked target speaker from first {reference_sec:.1f}s using similarity:",
                {k: round(v, 3) for k, v in scores.items()},
                flush=True,
            )
            return target, cluster_energy

    target = max(cluster_energy, key=cluster_energy.get)
    print("Selected loudest speaker by total energy.", flush=True)
    return target, cluster_energy


def build_speaker_mask(
    audio_len: int,
    sr: int,
    segments: list[tuple[int, int, float]],
    labels: np.ndarray,
    target: int,
    *,
    segment_pad_sec: float,
    gap_fill_sec: float,
    edge_pad_sec: float,
    fade_sec: float,
) -> np.ndarray:
    mask = np.zeros(audio_len, dtype=np.float32)
    pad = int(segment_pad_sec * sr)

    for (start, end, _), label in zip(segments, labels):
        if int(label) != target:
            continue
        s = max(0, start - pad)
        e = min(audio_len, end + pad)
        mask[s:e] = 1.0

    gap_fill = int(gap_fill_sec * sr)
    edge_pad = int(edge_pad_sec * sr)
    mask_bool = mask > 0.5
    mask_bool = binary_closing(mask_bool, structure=np.ones(gap_fill))
    mask_bool = binary_dilation(mask_bool, structure=np.ones(edge_pad))
    mask = mask_bool.astype(np.float32)

    fade = int(fade_sec * sr)
    kernel = np.ones(fade * 2 + 1) / (fade * 2 + 1)
    return np.clip(np.convolve(mask, kernel, mode="same"), 0.0, 1.0)


def isolate_target_speaker(
    in_path: Path,
    out_path: Path,
    inference: Inference,
    *,
    speaker_mode: str = "lock",
    reference_sec: float = 10.0,
    window_sec: float = 1.5,
    hop_sec: float = 0.2,
    min_rms: float = 0.004,
    max_speakers: int = 2,
    segment_pad_sec: float = 0.7,
    gap_fill_sec: float = 1.2,
    edge_pad_sec: float = 0.2,
    fade_sec: float = 0.06,
) -> None:
    audio, sr = sf.read(str(in_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    segments, embeddings = collect_segments(
        audio,
        sr,
        inference,
        window_sec=window_sec,
        hop_sec=hop_sec,
        min_rms=min_rms,
    )

    if len(embeddings) < 2:
        print("Warning: not enough speech for clustering; keeping vocals as-is.", flush=True)
        sf.write(str(out_path), audio, sr)
        return

    labels = AgglomerativeClustering(
        n_clusters=min(max_speakers, len(embeddings)),
        metric="cosine",
        linkage="average",
    ).fit_predict(np.vstack(embeddings))

    target, cluster_energy = choose_target_cluster(
        segments,
        embeddings,
        labels,
        speaker_mode=speaker_mode,
        reference_sec=reference_sec,
        sr=sr,
    )
    print("Speaker cluster energy:", cluster_energy, flush=True)
    print("Target speaker cluster:", target, flush=True)

    mask = build_speaker_mask(
        len(audio),
        sr,
        segments,
        labels,
        target,
        segment_pad_sec=segment_pad_sec,
        gap_fill_sec=gap_fill_sec,
        edge_pad_sec=edge_pad_sec,
        fade_sec=fade_sec,
    )

    isolated = audio * mask
    peak = np.max(np.abs(isolated))
    if peak > 0:
        isolated = isolated * (0.98 / peak)

    sf.write(str(out_path), isolated, sr)
    print(f"Saved speaker track: {out_path.name}", flush=True)


def process_video(
    input_video: Path,
    output_video: Path,
    hf_token: str,
    work_dir: Path | None = None,
    keep_intermediates: bool = False,
    speaker_mode: str = "lock",
    reference_sec: float = 10.0,
    window_sec: float = 1.5,
    hop_sec: float = 0.2,
    min_rms: float = 0.004,
    max_speakers: int = 2,
    segment_pad_sec: float = 0.7,
    gap_fill_sec: float = 1.2,
    edge_pad_sec: float = 0.2,
    fade_sec: float = 0.06,
) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed. Install it with: brew install ffmpeg")

    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is required. Set it in your environment or pass --hf-token.\n"
            "Get a token at https://huggingface.co/settings/tokens\n"
            "Accept model terms at https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM"
        )

    input_video = input_video.resolve()
    output_video = output_video.resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)

    temp_ctx = tempfile.TemporaryDirectory(prefix="remove_noise_")
    tmp = work_dir or Path(temp_ctx.name)
    tmp.mkdir(parents=True, exist_ok=True)

    raw_wav = tmp / "01_raw.wav"
    df1_wav = tmp / "02_denoised_pass1.wav"
    vocals_wav = tmp / "03_vocals.wav"
    speaker_wav = tmp / "04_dominant_speaker.wav"
    final_wav = tmp / "05_denoised_pass2.wav"
    demucs_dir = tmp / "demucs"

    try:
        print("\n[1/7] Extracting audio...", flush=True)
        extract_audio(input_video, raw_wav)

        print("\n[2/7] DeepFilterNet pass 1 (ambient noise)...", flush=True)
        deepfilter(raw_wav, df1_wav)

        print("\n[3/7] Demucs vocal isolation...", flush=True)
        vocals = demucs_vocals(df1_wav, demucs_dir)
        shutil.copy2(vocals, vocals_wav)

        print("\n[4/7] Loading Pyannote speaker model...", flush=True)
        inference = Inference(
            "pyannote/wespeaker-voxceleb-resnet34-LM",
            use_auth_token=hf_token,
        )

        print("\n[5/7] Isolating target speaker...", flush=True)
        isolate_target_speaker(
            vocals_wav,
            speaker_wav,
            inference,
            speaker_mode=speaker_mode,
            reference_sec=reference_sec,
            window_sec=window_sec,
            hop_sec=hop_sec,
            min_rms=min_rms,
            max_speakers=max_speakers,
            segment_pad_sec=segment_pad_sec,
            gap_fill_sec=gap_fill_sec,
            edge_pad_sec=edge_pad_sec,
            fade_sec=fade_sec,
        )

        print("\n[6/7] DeepFilterNet pass 2 (speech polish)...", flush=True)
        deepfilter(speaker_wav, final_wav)

        print("\n[7/7] Merging with video...", flush=True)
        merge_audio_video(input_video, final_wav, output_video)

        print(f"\nDone: {output_video}", flush=True)

        if keep_intermediates and work_dir is None:
            saved = input_video.parent / f"{input_video.stem}_intermediates"
            shutil.copytree(tmp, saved, dirs_exist_ok=True)
            print(f"Intermediates saved to: {saved}", flush=True)

    finally:
        if work_dir is None:
            temp_ctx.cleanup()


def prompt_input_path() -> Path:
    print("\n=== Remove Noise — Video Speaker Isolation ===\n")
    while True:
        path_str = input("Enter video file path: ").strip().strip('"').strip("'")
        if not path_str:
            print("Please enter a path.")
            continue
        path = Path(path_str).expanduser().resolve()
        if path.is_file():
            return path
        print(f"File not found: {path}")


def default_output_path(input_video: Path) -> Path:
    return input_video.with_name(f"{input_video.stem}_final.mp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove background noise and isolate a target speaker in a video.",
    )
    parser.add_argument(
        "input", type=Path, nargs="?", default=None,
        help="Input video file (prompted if omitted)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output video file (default: <input>_final.mp4)",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=None,
        help="Directory for intermediate files (default: temp folder)",
    )
    parser.add_argument(
        "--keep-intermediates", action="store_true",
        help="Save intermediate WAV files next to the input video",
    )
    parser.add_argument(
        "--speaker-mode",
        choices=("lock", "dominant"),
        default="lock",
        help="lock: keep speaker from first N seconds (default); dominant: keep loudest speaker overall",
    )
    parser.add_argument(
        "--reference-sec",
        type=float,
        default=10.0,
        help="Seconds at the start used to identify the target speaker (lock mode only)",
    )
    parser.add_argument("--window-sec", type=float, default=1.5, help="Speaker analysis window size")
    parser.add_argument("--hop-sec", type=float, default=0.2, help="Speaker analysis hop size")
    parser.add_argument("--min-rms", type=float, default=0.004, help="Ignore quieter audio below this level")
    parser.add_argument("--max-speakers", type=int, default=2, help="Maximum number of speaker clusters")
    parser.add_argument("--segment-pad-sec", type=float, default=0.7, help="Pad kept speech segments")
    parser.add_argument("--gap-fill-sec", type=float, default=1.2, help="Fill short silence gaps in speech")
    parser.add_argument("--edge-pad-sec", type=float, default=0.2, help="Extra mask expansion at edges")
    parser.add_argument("--fade-sec", type=float, default=0.06, help="Fade duration at mask boundaries")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_video = args.input.resolve() if args.input else prompt_input_path()
    output_video = args.output.resolve() if args.output else default_output_path(input_video)

    print(f"\nInput:  {input_video}")
    print(f"Output: {output_video}\n")

    process_video(
        input_video=input_video,
        output_video=output_video,
        hf_token=args.hf_token or "",
        work_dir=args.work_dir,
        keep_intermediates=args.keep_intermediates,
        speaker_mode=args.speaker_mode,
        reference_sec=args.reference_sec,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        min_rms=args.min_rms,
        max_speakers=args.max_speakers,
        segment_pad_sec=args.segment_pad_sec,
        gap_fill_sec=args.gap_fill_sec,
        edge_pad_sec=args.edge_pad_sec,
        fade_sec=args.fade_sec,
    )


if __name__ == "__main__":
    main()
