import os
import textwrap
from typing import Iterable, List

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .fs import ensure_dir

_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
)


def _load_font(size: int):
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except Exception:  # noqa: BLE001 — any font failure falls back
            continue
    # Bitmap fallback: tiny on a 960px composite, but never raises and never
    # leaves the caller without a font.
    return ImageFont.load_default()


def add_caption_band(
    frames: List[Image.Image], text: str, *, font_size: int = 18, max_lines: int = 2
) -> List[Image.Image]:
    """Prepend a text band ABOVE each frame and return the new frames.

    Burns the language instruction into the eval composite, so the clip is
    readable without hovering the wandb caption.

    The band is ADDED above rather than drawn over the image on purpose: the
    composite is a diagnostic grid whose top row is `pred_rgb / vae_recon /
    gt_rgb`, and drawing over it would hide predicted pixels — the one thing the
    panel exists to show. Height grows by ~30-50 px on a 2304 px composite.

    Returns `frames` unchanged when `text` is empty, so callers need no guard.
    """
    if not text:
        return frames
    frames = list(frames)
    if not frames:
        return frames

    w = frames[0].width
    font = _load_font(font_size)
    # Char-width estimate rather than per-line measurement: this is a caption
    # band, not typesetting, and it must not depend on Pillow's textlength API
    # (which moved between versions).
    max_chars = max(int(w / max(font_size * 0.55, 1.0)), 12)
    lines = textwrap.wrap(str(text).strip(), max_chars)[:max_lines] or [""]
    line_h = font_size + 4
    # Round the band to a multiple of 16 so the composite's height stays
    # macro-block aligned. Without this, a 2304 px composite becomes 2334 and
    # imageio silently resizes it to 2336 — a stretch the panel never had before.
    band_h = -(-(line_h * len(lines) + 8) // 16) * 16

    out = []
    for fr in frames:
        canvas = Image.new("RGB", (w, fr.height + band_h), (0, 0, 0))
        canvas.paste(fr, (0, band_h))
        draw = ImageDraw.Draw(canvas)
        for i, line in enumerate(lines):
            draw.text((6, 4 + i * line_h), line, fill=(255, 255, 255), font=font)
        out.append(canvas)
    return out


def _to_even_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def save_mp4(frames: Iterable[Image.Image], path: str, fps: int = 8):
    ensure_dir(os.path.dirname(path) or ".")
    # yuv444p (full chroma) instead of yuv420p (4:2:0 subsampled). 4:2:0
    # averages chroma in 2×1 column pairs, which is exactly the structure
    # produced by `tile_mode="repeat"` (np.repeat-style column doubling) —
    # so 4:2:0 silently turns a NN-doubled slot into a bilinearly-stretched
    # look in the saved MP4. yuv444p preserves
    # the duplication faithfully. Larger file, plays fine in modern players.
    writer = imageio.get_writer(
        path,
        fps=max(fps, 1),
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv444p",
    )
    try:
        for frame in frames:
            arr = np.array(frame.convert("RGB"))
            writer.append_data(_to_even_frame(arr))
    finally:
        writer.close()
