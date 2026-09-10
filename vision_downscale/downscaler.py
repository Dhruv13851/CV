"""Shrink oversized images to fit a vision model's input limit.

OpenAI tokenises images as 32x32 pixel patches and rejects anything above
30000 of them. A 600 DPI Letter scan (5100x6600) is 33120 patches and returns
HTTP 400 before the model ever runs. File size is irrelevant; only pixel
dimensions count.

Shared across services: `pip install` this package and instantiate
`ImageDownscaler`. The defaults are OpenAI's limits; pass your own to fit a
provider with different caps.
"""

import io
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

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
OPENAI_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "GIF"})

# Media types that go through downscale(). Anything else (PDF) is forwarded
# byte-for-byte.
DOWNSCALABLE = frozenset({"image/jpeg", "image/png", "image/heic", "image/heif"})

# Bounded, module-level, and shared by every instance that does not pass its
# own. Instances are cheap and a service may hold several profiles; a pool per
# instance would multiply 8 threads by the instance count, and a pool per
# request would grow threads without limit under load. 8 workers captured
# nearly all of the measured gain (12 pages: 0.42s at 8 workers, 0.40s at 12).
_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="downscale")

logger = logging.getLogger(__name__)

PATCH = 32
HARD_PATCH_LIMIT = 30000

# ponytail: 2200px long edge is ~200 DPI on a Letter page (3726 patches).
# Verified legible down to the footer fine print on samples/. Raise toward
# 3300 (300 DPI, the document-scanning standard) if a report with smaller
# type starts losing rows; lower it to cut image tokens.
MAX_LONG_EDGE = 2200

JPEG_QUALITY = 90


def _sizes_only(inputs: dict) -> dict:
    """Never let raw image bytes reach LangSmith.

    Returns a fresh dict rather than editing `inputs`, which also drops the
    bound `self` - serialising the instance would put nothing useful in the
    trace and is one more thing that could carry a payload.
    """
    data = inputs.get("file_bytes") or b""
    return {"input_bytes": len(data)}


def _result_summary(outputs) -> dict:
    if isinstance(outputs, tuple) and len(outputs) == 2:
        data, media_type = outputs
        return {"output_bytes": len(data), "media_type": media_type}
    return {"output": "unavailable"}


def _page(index: int, call, *args) -> tuple[bytes, str]:
    """One page's result, with any decode failure named by page number.

    Everything Pillow throws for an unreadable page is an OSError
    (UnidentifiedImageError for a mislabelled file, "image file is
    truncated" for a partial upload) except the decompression-bomb guard.
    The user cannot act on either message, but they can act on "page 7".
    """
    try:
        return call(*args)
    except (OSError, Image.DecompressionBombError) as exc:
        # ponytail: Pillow's own MAX_IMAGE_PIXELS is the bomb ceiling (~89M
        # pixels, hard error at 2x). Set it explicitly here if a legitimate
        # large-format scan ever trips it.
        logger.warning("page %d could not be decoded: %s", index + 1, exc)
        raise ValueError(
            f"Page {index + 1} could not be read as an image. "
            "Re-upload it as JPG, PNG or HEIC, or send the report as a PDF."
        ) from None


class ImageDownscaler:
    """Fit images under a vision model's input limits.

    One instance per provider profile. The defaults are OpenAI's: 32px
    patches, 30000 patches max per image, 2200px long edge.

        downscaler = ImageDownscaler()
        for file_bytes, media_type in downscaler.downscale_all(files):
            ...
    """

    def __init__(
        self,
        max_long_edge: int = MAX_LONG_EDGE,
        hard_patch_limit: int = HARD_PATCH_LIMIT,
        patch: int = PATCH,
        accepted_formats: frozenset[str] = OPENAI_FORMATS,
        downscalable: frozenset[str] = DOWNSCALABLE,
        jpeg_quality: int = JPEG_QUALITY,
        executor: ThreadPoolExecutor | None = None,
        profile: str = "openai",
    ):
        self.max_long_edge = max_long_edge
        self.hard_patch_limit = hard_patch_limit
        self.patch = patch
        self.accepted_formats = accepted_formats
        self.downscalable = downscalable
        self.jpeg_quality = jpeg_quality
        # Shared by default - see _POOL. Pass one only to isolate deliberately.
        self._executor = executor or _POOL
        # Names this profile in the trace, so two instances with different
        # caps are told apart in LangSmith. The span name itself cannot vary:
        # @traceable fixes it at decoration time.
        self.profile = profile

    def patch_count(self, width: int, height: int) -> int:
        """Image tokens the model will charge for these dimensions."""
        return math.ceil(width / self.patch) * math.ceil(height / self.patch)

    def needs_downscale(self, width: int, height: int) -> bool:
        return max(width, height) > self.max_long_edge or (
            self.patch_count(width, height) > self.hard_patch_limit
        )

    @traceable(
        run_type="tool",
        name="downscale_image",
        process_inputs=_sizes_only,
        process_outputs=_result_summary,
    )
    def downscale(self, file_bytes: bytes) -> tuple[bytes, str]:
        """Return (bytes, media_type), shrunk only if the image is oversized.

        Images already within the cap are returned untouched, so nothing small
        ever pays a lossy re-encode.
        """
        image = Image.open(io.BytesIO(file_bytes))
        source_format = image.format  # exif_transpose returns a copy, format None

        # 274 is the Orientation tag. Read it BEFORE transposing, because the
        # passthrough below returns the original bytes: an image small enough to
        # skip the resize would keep its sideways pixels, and we drop the tag on
        # every re-encode, so nothing downstream would ever turn it upright.
        exif_rotated = image.getexif().get(274, 1) != 1

        # Phones store landscape pixels plus an orientation tag. We re-encode
        # without that tag, so rotate the pixels now or the model reads a
        # portrait report sideways. No-op for the tagless and already-upright.
        image = ImageOps.exif_transpose(image)

        width, height = image.size

        run = get_current_run_tree()
        if run is not None:
            run.metadata.update({
                "profile": self.profile,
                "source_format": source_format,
                "source_mode": image.mode,
                "width_before": width,
                "height_before": height,
                "patches_before": self.patch_count(width, height),
                "patch_limit": self.hard_patch_limit,
                "long_edge_cap": self.max_long_edge,
                "exif_rotated": exif_rotated,
            })

        if (
            not exif_rotated
            and not self.needs_downscale(width, height)
            and source_format in self.accepted_formats
        ):
            if run is not None:
                run.metadata["resized"] = False
            return file_bytes, Image.MIME.get(source_format, "image/png")

        if self.needs_downscale(width, height):
            scale = self.max_long_edge / max(width, height)
            target = (max(1, round(width * scale)), max(1, round(height * scale)))
        else:
            # A HEIC already under the cap still has to be transcoded. Keep its
            # pixels; upscaling to max_long_edge would invent detail.
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
            quality=self.jpeg_quality,
            subsampling=0,  # 4:4:4 - keeps coloured flags and thin rules crisp
            optimize=True,
        )

        if run is not None:
            run.metadata.update({
                "resized": target != (width, height),
                "width_after": target[0],
                "height_after": target[1],
                "patches_after": self.patch_count(*target),
                "patch_reduction": round(
                    self.patch_count(width, height) / self.patch_count(*target), 1
                ),
            })

        return buffer.getvalue(), "image/jpeg"

    def _prepare(self, entry: tuple[bytes, str]) -> tuple[bytes, str]:
        """One page, ready to send. Runs in a pool thread."""
        file_bytes, media_type = entry

        if media_type in self.downscalable:
            return self.downscale(file_bytes)

        return entry

    def downscale_all(
        self,
        files: list[tuple[bytes, str]],
    ) -> list[tuple[bytes, str]]:
        """Prepare every page in parallel, preserving upload order.

        Measured on 12 phone HEICs: 9.78s serial -> 1.95s pooled, 5.0x. Pillow
        releases the GIL during decode, resize and encode, so threads genuinely
        parallelise this. It matters for phone uploads specifically: HEVC decode
        puts one HEIC page at 844ms against 97ms for an already-small JPEG, so a
        12-page HEIC batch is ~8s of the request rather than ~1s.

        Each page gets its own copy_context() because ThreadPoolExecutor does not
        propagate contextvars the way asyncio.to_thread does - without it the
        @traceable downscale_image spans lose their parent and show up in
        LangSmith as orphan roots. One Context cannot be entered from two threads
        at once, hence a copy per page rather than one shared copy.
        """
        # ponytail: no pool for a single page - a PDF or one photo is the common
        # case and there is nothing to overlap.
        if len(files) < 2:
            return [_page(i, self._prepare, entry) for i, entry in enumerate(files)]

        futures = [
            self._executor.submit(copy_context().run, self._prepare, entry)
            for entry in files
        ]

        return [_page(i, future.result) for i, future in enumerate(futures)]
