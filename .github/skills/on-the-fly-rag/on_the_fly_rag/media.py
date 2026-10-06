"""Image / video / audio discovery and decoding for multimodal ingest.

Only used with a multimodal embedder (``embeddinggemma-2``). Decoding uses
``ffmpeg`` (system binary, or the one bundled by ``imageio-ffmpeg``) so no
torchcodec/av dependency is required. Images use Pillow.

Each media file becomes one or more :class:`MediaItem` segments:

* image  → 1 item (GIF: first frame)
* video  → frames sampled at ``fps`` (default 1), grouped into segments of
  ``segment_frames`` frames (default 16 → 16 s @1fps → 16×140 = 2,240 vision
  tokens, well under the 8,192 shared context and the processor's 32-frame cap)
* audio  → 16 kHz mono windows of ``window_s`` seconds (default 10 s; the
  shipped processor caps one audio item at 280 soft tokens ≈ 11.2 s)
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm")
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS + AUDIO_EXTENSIONS

# Token accounting from the EmbeddingGemma 2 model card / processor config.
CONTEXT_TOKENS = 8192
IMAGE_TOKENS = 280
VIDEO_FRAME_TOKENS = 140
AUDIO_TOKENS_PER_S = 25
PROCESSOR_MAX_VIDEO_FRAMES = 32  # processor_config.json video_processor.max_frames
PROCESSOR_MAX_AUDIO_TOKENS = 280  # processor_config.json audio_seq_length
AUDIO_SR = 16000

DEFAULT_VIDEO_FPS = 1.0
DEFAULT_SEGMENT_FRAMES = 16
DEFAULT_MAX_VIDEO_FRAMES = 600  # total per file (10 min @1fps)
DEFAULT_AUDIO_WINDOW_S = 10.0
DEFAULT_MAX_AUDIO_S = 1800.0
MIN_AUDIO_S = 0.05
FRAME_MAX_SIDE = 768  # downscale decoded frames (processor resizes anyway)


def modality_of(path: Path | str) -> str:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    return "text"


def is_media(path: Path | str) -> bool:
    return modality_of(path) != "text"


def fmt_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    if h:
        return f"{h:d}:{m:02d}:{s:04.1f}"
    return f"{m:02d}:{s:04.1f}"


@dataclass
class MediaItem:
    """One embeddable media segment."""

    path: str  # relative display path incl. fragment (e.g. clip.mp4#t=16.0-32.0)
    source: str  # relative source file path (no fragment)
    modality: str  # image | video | audio
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None
    description: str = ""
    payload: Any = field(default=None, repr=False)  # model input (PIL / ndarray)
    fps: Optional[float] = None  # video sampling rate

    def model_input(self) -> Any:
        if self.modality == "image":
            return self.payload
        if self.modality == "audio":
            return {"array": self.payload, "sampling_rate": AUDIO_SR}
        if self.modality == "video":
            frames = self.payload
            n = len(frames)
            fps = self.fps or DEFAULT_VIDEO_FPS
            return {
                "array": frames,
                "video_metadata": {
                    "fps": fps,
                    "total_num_frames": n,
                    "duration": n / fps,
                },
            }
        raise ValueError(self.modality)


# --------------------------------------------------------------------------- ffmpeg


def ffmpeg_exe() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def require_ffmpeg() -> str:
    exe = ffmpeg_exe()
    if not exe:
        raise RuntimeError(
            "ffmpeg not found. Install it (apt-get install ffmpeg / brew install ffmpeg) "
            "or `pip install imageio-ffmpeg` (included in requirements-gemma.txt)."
        )
    return exe


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VID_RE = re.compile(r"Stream #.*Video:.*?(\d{2,5})x(\d{2,5})")


def probe(path: Path | str) -> Dict[str, Any]:
    """Return {duration, width, height, has_video, has_audio} (best effort)."""
    path = str(path)
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            out = subprocess.run(
                [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
                capture_output=True,
                check=True,
                timeout=60,
            ).stdout
            meta = json.loads(out or b"{}")
            streams = meta.get("streams", [])
            v = next((s for s in streams if s.get("codec_type") == "video"), None)
            a = next((s for s in streams if s.get("codec_type") == "audio"), None)
            dur = meta.get("format", {}).get("duration")
            return {
                "duration": float(dur) if dur not in (None, "N/A") else None,
                "width": int(v["width"]) if v and v.get("width") else None,
                "height": int(v["height"]) if v and v.get("height") else None,
                "has_video": v is not None,
                "has_audio": a is not None,
            }
        except (subprocess.SubprocessError, ValueError, OSError) as e:
            logger.debug("ffprobe failed for %s: %s", path, e)
    exe = require_ffmpeg()
    res = subprocess.run([exe, "-hide_banner", "-i", path], capture_output=True, timeout=60)
    err = res.stderr.decode("utf-8", "replace")
    m = _DUR_RE.search(err)
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None
    vm = _VID_RE.search(err)
    return {
        "duration": dur,
        "width": int(vm.group(1)) if vm else None,
        "height": int(vm.group(2)) if vm else None,
        "has_video": "Video:" in err,
        "has_audio": "Audio:" in err,
    }


def decode_audio(path: Path | str, *, max_seconds: Optional[float] = None) -> np.ndarray:
    """Decode any audio (or a video's audio track) to 16 kHz mono float32."""
    exe = require_ffmpeg()
    cmd = [exe, "-v", "error", "-i", str(path)]
    if max_seconds:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(AUDIO_SR), "-f", "f32le", "-"]
    res = subprocess.run(cmd, capture_output=True, timeout=1800)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg audio decode failed for {path}: {res.stderr.decode()[-400:]}")
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


def decode_video_frames(
    path: Path | str,
    *,
    fps: float = DEFAULT_VIDEO_FPS,
    max_frames: int = DEFAULT_MAX_VIDEO_FRAMES,
    max_side: int = FRAME_MAX_SIDE,
) -> np.ndarray:
    """Return uint8 RGB frames (N, H, W, 3) sampled at ``fps`` (capped)."""
    info = probe(path)
    w, h = info.get("width"), info.get("height")
    if not w or not h:
        raise RuntimeError(f"no video stream in {path}")
    scale = min(1.0, max_side / float(max(w, h)))
    ow, oh = max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)
    exe = require_ffmpeg()
    cmd = [
        exe, "-v", "error", "-i", str(path),
        "-vf", f"fps={fps},scale={ow}:{oh}",
        "-frames:v", str(int(max_frames)),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    res = subprocess.run(cmd, capture_output=True, timeout=1800)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg video decode failed for {path}: {res.stderr.decode()[-400:]}")
    frame_bytes = ow * oh * 3
    n = len(res.stdout) // frame_bytes
    if n == 0:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.frombuffer(res.stdout[: n * frame_bytes], dtype=np.uint8).reshape(n, oh, ow, 3).copy()


def load_image(path: Path | str):
    from PIL import Image

    im = Image.open(str(path))
    try:
        im.seek(0)  # GIF / animated WebP → first frame
    except EOFError:
        pass
    return im.convert("RGB")


# --------------------------------------------------------------------------- segmenting


@dataclass
class MediaOptions:
    video_fps: float = DEFAULT_VIDEO_FPS
    segment_frames: int = DEFAULT_SEGMENT_FRAMES
    max_video_frames: int = DEFAULT_MAX_VIDEO_FRAMES
    audio_window_s: float = DEFAULT_AUDIO_WINDOW_S
    max_audio_s: float = DEFAULT_MAX_AUDIO_S

    def validated(self) -> "MediaOptions":
        if self.video_fps <= 0:
            raise ValueError("video fps must be > 0")
        seg = int(self.segment_frames)
        if seg < 1:
            raise ValueError("segment_frames must be >= 1")
        # Stay inside the processor's per-video frame cap and the shared context.
        seg = min(seg, PROCESSOR_MAX_VIDEO_FRAMES, (CONTEXT_TOKENS - 64) // VIDEO_FRAME_TOKENS)
        win = float(self.audio_window_s)
        max_win = PROCESSOR_MAX_AUDIO_TOKENS / AUDIO_TOKENS_PER_S  # 11.2 s
        if win <= 0:
            raise ValueError("audio window must be > 0")
        win = min(win, max_win)
        return MediaOptions(
            video_fps=float(self.video_fps),
            segment_frames=seg,
            max_video_frames=int(self.max_video_frames),
            audio_window_s=win,
            max_audio_s=float(self.max_audio_s),
        )


def media_items_for(path: Path, *, root: Path, opts: Optional[MediaOptions] = None) -> List[MediaItem]:
    """Decode ``path`` and split it into embeddable segments (raises on failure)."""
    opts = (opts or MediaOptions()).validated()
    rel = str(path.resolve().relative_to(root.resolve()))
    mod = modality_of(path)
    name = path.name
    if mod == "image":
        img = load_image(path)
        return [
            MediaItem(
                path=rel, source=rel, modality="image",
                description=f"[image] {name} ({img.width}x{img.height})",
                payload=img,
            )
        ]
    if mod == "audio":
        wave = decode_audio(path, max_seconds=opts.max_audio_s)
        return _audio_windows(wave, rel=rel, name=name, window_s=opts.audio_window_s)
    if mod == "video":
        frames = decode_video_frames(path, fps=opts.video_fps, max_frames=opts.max_video_frames)
        items: List[MediaItem] = []
        seg = opts.segment_frames
        bounds = [(i, min(i + seg, len(frames))) for i in range(0, len(frames), seg)]
        # Fold a tiny tail (< 1/4 segment) into the previous segment when it still fits.
        if len(bounds) > 1:
            (a, _), (_, z) = bounds[-2], bounds[-1]
            tail = bounds[-1][1] - bounds[-1][0]
            if tail < max(2, seg // 4) and z - a <= PROCESSOR_MAX_VIDEO_FRAMES:
                bounds[-2:] = [(a, z)]
        for i, j in bounds:
            chunk = frames[i:j]
            t0, t1 = i / opts.video_fps, (i + len(chunk)) / opts.video_fps
            it = MediaItem(
                path=f"{rel}#t={t0:.1f}-{t1:.1f}",
                source=rel, modality="video",
                start_sec=round(t0, 3), end_sec=round(t1, 3),
                frame_start=i, frame_end=i + len(chunk) - 1,
                description=(
                    f"[video] {name} {fmt_ts(t0)}–{fmt_ts(t1)} "
                    f"({len(chunk)} frames @{opts.video_fps:g}fps)"
                ),
                payload=chunk,
                fps=opts.video_fps,
            )
            items.append(it)
        return items
    raise ValueError(f"not a media file: {path}")


def _audio_windows(wave: np.ndarray, *, rel: str, name: str, window_s: float) -> List[MediaItem]:
    total_s = len(wave) / AUDIO_SR
    if total_s < MIN_AUDIO_S:
        raise RuntimeError(f"audio too short ({total_s:.3f}s)")
    n_win = max(1, math.ceil(total_s / window_s - 1e-6))
    win = int(window_s * AUDIO_SR)
    items: List[MediaItem] = []
    for k in range(n_win):
        a = wave[k * win : (k + 1) * win]
        if k > 0 and len(a) < MIN_AUDIO_S * AUDIO_SR * 10:  # drop <0.5 s tails
            break
        t0, t1 = k * window_s, min(total_s, (k + 1) * window_s)
        items.append(
            MediaItem(
                path=f"{rel}#t={t0:.1f}-{t1:.1f}" if n_win > 1 else rel,
                source=rel, modality="audio",
                start_sec=round(t0, 3), end_sec=round(t1, 3),
                description=f"[audio] {name} {fmt_ts(t0)}–{fmt_ts(t1)}",
                payload=a,
            )
        )
    return items


def query_input_for_file(path: Path | str, opts: Optional[MediaOptions] = None) -> Tuple[str, Any]:
    """Model input for a media *query* file (first segment/window only)."""
    path = Path(path)
    items = media_items_for(path, root=path.parent, opts=opts)
    return items[0].modality, items[0].model_input()


def iter_media(files: List[Path]) -> Iterator[Path]:
    for f in files:
        if is_media(f):
            yield f
