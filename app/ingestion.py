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