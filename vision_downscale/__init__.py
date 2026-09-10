from vision_downscale.downscaler import (
    DOWNSCALABLE,
    HARD_PATCH_LIMIT,
    JPEG_QUALITY,
    MAX_LONG_EDGE,
    OPENAI_FORMATS,
    PATCH,
    ImageDownscaler,
)

# The default profile. Services that want OpenAI's limits import this (or the
# functions below) instead of building their own instance.
OPENAI = ImageDownscaler()

# Module-level shorthand, bound to OPENAI. Callers that never needed a second
# profile keep working unchanged.
downscale = OPENAI.downscale
downscale_all = OPENAI.downscale_all
patch_count = OPENAI.patch_count
needs_downscale = OPENAI.needs_downscale

__all__ = [
    "DOWNSCALABLE",
    "HARD_PATCH_LIMIT",
    "JPEG_QUALITY",
    "MAX_LONG_EDGE",
    "OPENAI",
    "OPENAI_FORMATS",
    "PATCH",
    "ImageDownscaler",
    "downscale",
    "downscale_all",
    "needs_downscale",
    "patch_count",
]
