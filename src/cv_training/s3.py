"""Create an S3 client only when an S3 backend is actually constructed."""

from __future__ import annotations


def create_s3_client():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "S3 support requires boto3. Install it with: pip install 'cv-training-service[s3]'"
        ) from exc
    return boto3.client("s3")


def is_not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    code = str(response.get("Error", {}).get("Code", ""))
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"404", "NoSuchKey", "NotFound"} or status == 404


def join_key(*parts: str) -> str:
    cleaned = [part.strip("/") for part in parts if part and part.strip("/")]
    return "/".join(cleaned)
