"""Shrink oversized images to fit OpenAI's vision input limit.

OpenAI tokenises images as 32x32 pixel patches and rejects anything above
30000 of them. A 600 DPI Letter scan (5100x6600) is 33120 patches and returns
HTTP 400 before the model ever runs. File size is irrelevant; only pixel
dimensions count.
"""

import io
import math

from langsmith import get_current_run_tree, traceable
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

# iPhones and recent Android shoot HEIC by default. Pillow ships no HEIF
# decoder (libheif is not bundled), and OpenAI's vision endpoint does not
# accept HEIC either - so it has to be decoded here and sent as JPEG.
# One call, and Image.open() handles .heic/.heif like any other format.
register_heif_opener()

# What OpenAI will actually accept. Anything else must be re-encoded even
# when it is small enough to skip resizing.
OPENAI_FORMATS = {"JPEG", "PNG", "WEBP", "GIF"}

PATCH = 32
HARD_PATCH_LIMIT = 30000

# ponytail: 2200px long edge is ~200 DPI on a Letter page (3726 patches).
# Verified legible down to the footer fine print on samples/. Raise toward
# 3300 (300 DPI, the document-scanning standard) if a report with smaller
# type starts losing rows; lower it to cut image tokens.
MAX_LONG_EDGE = 2200

JPEG_QUALITY = 90


def patch_count(width: int, height: int) -> int:
    """Image tokens OpenAI will charge for these dimensions."""
    return math.ceil(width / PATCH) * math.ceil(height / PATCH)


def needs_downscale(width: int, height: int) -> bool:
    return max(width, height) > MAX_LONG_EDGE or (
        patch_count(width, height) > HARD_PATCH_LIMIT
    )


def _sizes_only(inputs: dict) -> dict:
    """Never let raw image bytes reach LangSmith."""
    data = inputs.get("file_bytes") or b""
    return {"input_bytes": len(data)}


def _result_summary(outputs) -> dict:
    if isinstance(outputs, tuple) and len(outputs) == 2:
        data, media_type = outputs
        return {"output_bytes": len(data), "media_type": media_type}
    return {"output": "unavailable"}


@traceable(
    run_type="tool",
    name="downscale_image",
    process_inputs=_sizes_only,
    process_outputs=_result_summary,
)
def downscale(file_bytes: bytes) -> tuple[bytes, str]:
    """Return (bytes, media_type), shrunk only if the image is oversized.

    Images already within the cap are returned untouched, so nothing small
    ever pays a lossy re-encode.
    """
    image = Image.open(io.BytesIO(file_bytes))
    source_format = image.format  # exif_transpose returns a copy, format None

    # Phones store landscape pixels plus an orientation tag. We re-encode
    # without that tag, so rotate the pixels now or the model reads a
    # portrait report sideways. No-op for the tagless and already-upright.
    image = ImageOps.exif_transpose(image)

    width, height = image.size

    run = get_current_run_tree()
    if run is not None:
        run.metadata.update({
            "source_format": source_format,
            "source_mode": image.mode,
            "width_before": width,
            "height_before": height,
            "patches_before": patch_count(width, height),
            "patch_limit": HARD_PATCH_LIMIT,
            "long_edge_cap": MAX_LONG_EDGE,
        })

    if not needs_downscale(width, height) and source_format in OPENAI_FORMATS:
        if run is not None:
            run.metadata["resized"] = False
        return file_bytes, Image.MIME.get(source_format, "image/png")

    if needs_downscale(width, height):
        scale = MAX_LONG_EDGE / max(width, height)
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
    else:
        # A HEIC already under the cap still has to be transcoded. Keep its
        # pixels; upscaling to MAX_LONG_EDGE would invent detail.
        target = (width, height)

    # Flatten onto white first: JPEG cannot hold an alpha channel, and
    # scans arrive as RGBA/P/CMYK often enough to matter.
    if image.mode != "RGB":
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            canvas = Image.new("RGB", image.size, (255, 255, 255))
            canvas.paste(image, mask=image.split()[-1])
            image = canvas
        else:
            image = image.convert("RGB")

    # LANCZOS preserves thin strokes; bilinear/nearest smear digits into
    # each other, which is how you turn an 8 into a 3.
    resized = image if target == image.size else image.resize(target, Image.LANCZOS)

    buffer = io.BytesIO()
    resized.save(
        buffer,
        format="JPEG",
        quality=JPEG_QUALITY,
        subsampling=0,  # 4:4:4 - keeps coloured flags and thin rules crisp
        optimize=True,
    )

    if run is not None:
        run.metadata.update({
            "resized": target != (width, height),
            "width_after": target[0],
            "height_after": target[1],
            "patches_after": patch_count(*target),
            "patch_reduction": round(
                patch_count(width, height) / patch_count(*target), 1
            ),
        })

    return buffer.getvalue(), "image/jpeg"
