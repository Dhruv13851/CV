from pathlib import Path


SUPPORTED_TYPES = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".heic": "image/heic",
    ".heif": "image/heif",
}


def get_media_type(filename: str) -> str:
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_TYPES:
        raise ValueError(
            "Unsupported file type. "
            "Only PDF, JPG, JPEG, PNG, HEIC, and HEIF are supported."
        )

    return SUPPORTED_TYPES[extension]


# One upload is one report. 12 photographed pages is roughly 52k image tokens
# at MAX_LONG_EDGE=2200 (3588 patches x 1.2 each). OpenAI's own ceilings are
# 30000 patches PER IMAGE, 1500 images and 512 MB per request - all far above
# this, so these two are spend caps, not API limits.
MAX_PAGES = 12
MAX_UPLOAD_BYTES = 30 * 1024 * 1024


def media_types_for(filenames: list[str]) -> list[str]:
    """Validate a whole upload: one PDF, or up to MAX_PAGES images.

    Returns one media type per name, in the order given - that order is the
    page order, so never sort it (IMG_9 sorts after IMG_10).
    """
    if not filenames:
        raise ValueError("No file uploaded.")

    media_types = [get_media_type(name) for name in filenames]

    if "application/pdf" in media_types and len(media_types) > 1:
        raise ValueError(
            "Upload one PDF, or photos of the pages - not both."
        )

    if len(media_types) > MAX_PAGES:
        raise ValueError(
            f"Too many pages: {len(media_types)}. "
            f"The limit is {MAX_PAGES}."
        )

    return media_types


def read_file(path: str) -> tuple[bytes, str]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    media_type = get_media_type(file_path.name)
    file_bytes = file_path.read_bytes()

    if not file_bytes:
        raise ValueError("File is empty.")

    return file_bytes, media_type