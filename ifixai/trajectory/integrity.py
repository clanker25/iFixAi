"""Portable checksummed audit evidence; integrity is not authenticity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .models import AuditBundle, AuditPolicy, TraceCapture


def canonical_digest(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


arguments_digest = canonical_digest


def create_audit_bundle(capture: TraceCapture, policy: AuditPolicy) -> AuditBundle:
    from .audit import audit_trajectory

    payload = {"schema_version": "1", "policy": policy.model_dump(mode="json"),
               "capture": capture.model_dump(mode="json"),
               "report": audit_trajectory(capture, policy).model_dump(mode="json")}
    return AuditBundle.model_validate({**payload, "integrity_digest": canonical_digest(payload)})


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError("duplicate JSON keys are not accepted")
        value[key] = child
    return value


def _reject_constant(_: str) -> None:
    raise ValueError("nonfinite JSON constants are not accepted")


def load_audit_bundle(path: str | Path) -> AuditBundle:
    from .audit import audit_trajectory

    source = Path(path)
    limit = 256 * 1024 * 1024
    with source.open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError("audit bundle exceeds the 256 MiB replay limit")
    raw = json.loads(content.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    bundle = AuditBundle.model_validate(raw)
    normalized = bundle.model_dump(mode="json")
    if normalized != raw:
        raise ValueError("audit bundle must contain complete normalized evidence")
    if canonical_digest(bundle.model_dump(mode="json", exclude={"integrity_digest"})) != bundle.integrity_digest:
        raise ValueError("audit bundle integrity digest mismatch")
    # Releases up to 4.2.0 stored a bad grant's finding once per use. An exact repeat adds
    # no fact, so drop repeats before comparing; any changed finding still mismatches.
    stored = bundle.report.model_copy(update={"violations": tuple(dict.fromkeys(bundle.report.violations))})
    if audit_trajectory(bundle.capture, bundle.policy) != stored:
        raise ValueError("audit report does not match the retained policy and capture")
    return bundle
