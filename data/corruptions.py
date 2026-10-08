from __future__ import annotations

import argparse
import io
import json
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from prodml.config import settings
from prodml.schemas import ManifestRecord

SEVERITIES: tuple[int, ...] = (1, 2, 3)
RESAMPLE = Image.Resampling.BILINEAR

BLUR_RADIUS = (1, 2, 4)
BRIGHTNESS_UP = (1.3, 1.6, 2.0)
BRIGHTNESS_DOWN = (0.7, 0.5, 0.3)
JPEG_QUALITY = (60, 30, 10) 
DOWNSCALE_SIDE = (160, 96, 64)  
MOTION_LENGTH = (5, 9, 15)


def _level[T](params: tuple[T, ...], severity: int) -> T:
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be in {SEVERITIES}, got {severity}")
    return params[severity - 1]


def gaussian_blur(img: Image.Image, severity: int) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=_level(BLUR_RADIUS, severity)))


def brightness_up(img: Image.Image, severity: int) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(_level(BRIGHTNESS_UP, severity))


def brightness_down(img: Image.Image, severity: int) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(_level(BRIGHTNESS_DOWN, severity))


def jpeg(img: Image.Image, severity: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=_level(JPEG_QUALITY, severity))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def downscale(img: Image.Image, severity: int) -> Image.Image:
    side = _level(DOWNSCALE_SIDE, severity)
    small = img.resize((side, side), RESAMPLE)
    return small.resize(img.size, RESAMPLE)


def motion_blur(img: Image.Image, severity: int) -> Image.Image:
    k = _level(MOTION_LENGTH, severity)
    arr = np.asarray(img, dtype=np.float32)
    width = arr.shape[1]
    padded = np.pad(arr, ((0, 0), (k // 2, k // 2), (0, 0)), mode="edge")
    out = np.mean([padded[:, i : i + width] for i in range(k)], axis=0)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


CORRUPTIONS: dict[str, Callable[[Image.Image, int], Image.Image]] = {
    "gaussian_blur": gaussian_blur,
    "brightness_up": brightness_up,
    "brightness_down": brightness_down,
    "jpeg": jpeg,
    "downscale": downscale,
    "motion_blur": motion_blur,
}


def get_corruption_names() -> list[str]:
    return list(CORRUPTIONS)


def corrupt(image: Image.Image, name: str, severity: int) -> Image.Image:
    """Apply corruption `name` at `severity` (1 to 3). Always returns an RGB image."""
    if name not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {name!r}, choose from {get_corruption_names()}")
    return CORRUPTIONS[name](image.convert("RGB"), severity)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate corrupted copies of the test set.")
    p.add_argument("--corruption", nargs="+", choices=get_corruption_names(),
                   default=get_corruption_names(), help="corruptions to apply (default: all)")
    p.add_argument("--severity", nargs="+", type=int, choices=SEVERITIES,
                   default=list(SEVERITIES), help="severity levels (default: all)")
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N test images (quick check)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(settings.manifest_path.read_text())
    test = [
        ManifestRecord(**r)
        for r in manifest
        if r["split"] == "test" and r["corruption"] is None
    ]
    if args.limit is not None:
        test = test[: args.limit]

    records: list[ManifestRecord] = []
    for name in args.corruption:
        for severity in args.severity:
            folder = settings.corrupted_root / name / f"s{severity}"
            folder.mkdir(parents=True, exist_ok=True)
            for rec in test:
                with Image.open(rec.path) as im:
                    out = corrupt(im, name, severity)
                path = folder / f"{rec.image_id}.jpg"
                out.save(path, "JPEG", quality=95)
                records.append(
                    rec.model_copy(
                        update={
                            "path": path.as_posix(),
                            "corruption": name,
                            "severity": severity,
                            "width": out.width,
                            "height": out.height,
                        }
                    )
                )
            print(f"{name} s{severity}: {len(test)} images")

    settings.corrupted_manifest_path.write_text(
        json.dumps([r.model_dump() for r in records], indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()