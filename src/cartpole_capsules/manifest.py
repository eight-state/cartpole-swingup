"""Strict manifest model for immutable capsule imports."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from cartpole_capsules.layout import manifest_path

SCHEMA_VERSION = 1
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SLUG = re.compile(r"^n(?P<rung>[0-9]{2})-[a-z0-9]+(?:-[a-z0-9]+)*$")
_TOKEN = re.compile(r"^[a-z0-9]+(?:[+-][a-z0-9]+)*$")
_RUNNERS = {"ubuntu-latest", "windows-latest"}
_FIELDS = {
    "rung",
    "slug",
    "source_url",
    "source_commit",
    "source_tree",
    "capsule_path",
    "content_hash",
    "file_count",
    "verification_command",
    "demo_command",
    "runner",
    "evidence_class",
}


class ManifestError(ValueError):
    """Raised when a manifest or entry violates the registry contract."""


@dataclass(frozen=True)
class CapsuleEntry:
    """One immutable capsule import."""

    rung: int
    slug: str
    source_url: str
    source_commit: str
    source_tree: str
    capsule_path: str
    content_hash: str
    file_count: int
    verification_command: str
    demo_command: str | None
    runner: str
    evidence_class: str

    @classmethod
    def from_mapping(cls, value: object) -> CapsuleEntry:
        """Validate and construct an entry from decoded JSON."""
        if not isinstance(value, dict):
            raise ManifestError("each capsule entry must be a JSON object")
        keys = set(value)
        missing = sorted(_FIELDS - keys)
        unknown = sorted(keys - _FIELDS)
        if missing:
            raise ManifestError(f"entry is missing fields: {', '.join(missing)}")
        if unknown:
            raise ManifestError(f"entry has unknown fields: {', '.join(unknown)}")

        rung = value["rung"]
        if isinstance(rung, bool) or not isinstance(rung, int) or not 5 <= rung <= 15:
            raise ManifestError("rung must be an integer from 5 through 15")

        slug = _text(value["slug"], "slug")
        slug_match = _SLUG.fullmatch(slug)
        if slug_match is None:
            raise ManifestError("slug must match nNN-lowercase-name")
        if int(slug_match.group("rung")) != rung:
            raise ManifestError("slug rung must match rung")

        source_url = _text(value["source_url"], "source_url")
        _validate_source_url(source_url)
        source_commit = _hex(value["source_commit"], "source_commit", _SHA1)
        source_tree = _hex(value["source_tree"], "source_tree", _SHA1)
        capsule_path = _text(value["capsule_path"], "capsule_path")
        _validate_capsule_path(capsule_path, slug)
        content_hash = _hex(value["content_hash"], "content_hash", _SHA256)

        file_count = value["file_count"]
        if isinstance(file_count, bool) or not isinstance(file_count, int) or file_count < 1:
            raise ManifestError("file_count must be an integer of at least 1")

        verification_command = _command(value["verification_command"], "verification_command")
        demo_value = value["demo_command"]
        demo_command = None if demo_value is None else _command(demo_value, "demo_command")

        runner = _text(value["runner"], "runner")
        if runner not in _RUNNERS:
            raise ManifestError(f"runner must be one of: {', '.join(sorted(_RUNNERS))}")
        evidence_class = _text(value["evidence_class"], "evidence_class")
        if _TOKEN.fullmatch(evidence_class) is None:
            raise ManifestError("evidence_class must be a lowercase token joined by + or -")

        return cls(
            rung=rung,
            slug=slug,
            source_url=source_url,
            source_commit=source_commit,
            source_tree=source_tree,
            capsule_path=capsule_path,
            content_hash=content_hash,
            file_count=file_count,
            verification_command=verification_command,
            demo_command=demo_command,
            runner=runner,
            evidence_class=evidence_class,
        )

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-ready mapping with stable field names."""
        return asdict(self)


def load_manifest(root: Path) -> list[CapsuleEntry]:
    """Read and validate the repository manifest."""
    path = manifest_path(root)
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "capsules"}:
        raise ManifestError("manifest must contain only schema_version and capsules")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported schema_version {raw['schema_version']!r}; expected {SCHEMA_VERSION}"
        )
    if not isinstance(raw["capsules"], list):
        raise ManifestError("capsules must be a JSON array")

    entries = [CapsuleEntry.from_mapping(item) for item in raw["capsules"]]
    _validate_collection(entries)
    return entries


def write_manifest(root: Path, entries: list[CapsuleEntry]) -> None:
    """Validate and atomically write a sorted manifest."""
    entries = sorted(entries, key=lambda entry: entry.rung)
    _validate_collection(entries)
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "capsules": [entry.to_mapping() for entry in entries],
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def capsule_directory(root: Path, entry: CapsuleEntry) -> Path:
    """Resolve a validated capsule path below the repository root."""
    root = root.resolve()
    candidate = (root / PurePosixPath(entry.capsule_path)).resolve()
    if root not in candidate.parents:
        raise ManifestError(f"capsule path escapes repository root: {entry.capsule_path}")
    return candidate


def _validate_collection(entries: list[CapsuleEntry]) -> None:
    for field in ("rung", "slug", "capsule_path"):
        values = [getattr(entry, field) for entry in entries]
        if len(values) != len(set(values)):
            raise ManifestError(f"duplicate {field} in manifest")
    rungs = [entry.rung for entry in entries]
    if rungs != sorted(rungs):
        raise ManifestError("capsules must be sorted by rung")


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ManifestError(f"{field} must be a non-empty trimmed string")
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise ManifestError(f"{field} must not contain control characters")
    return value


def _command(value: object, field: str) -> str:
    command = _text(value, field)
    if not command.startswith("uv run "):
        raise ManifestError(f"{field} must start with 'uv run '")
    return command


def _hex(value: object, field: str, pattern: re.Pattern[str]) -> str:
    text = _text(value, field)
    if pattern.fullmatch(text) is None:
        bits = 40 if pattern is _SHA1 else 64
        raise ManifestError(f"{field} must be exactly {bits} lowercase hexadecimal characters")
    return text


def _validate_source_url(value: str) -> None:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ManifestError("source_url must be a plain HTTPS github.com repository URL")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2 or any(part in {".", ".."} for part in parts):
        raise ManifestError("source_url must identify one GitHub owner and repository")


def _validate_capsule_path(value: str, slug: str) -> None:
    if "\\" in value:
        raise ManifestError("capsule_path must use forward slashes")
    path = PurePosixPath(value)
    if path.is_absolute() or path.parts != ("capsules", slug):
        raise ManifestError(f"capsule_path must equal capsules/{slug}")
