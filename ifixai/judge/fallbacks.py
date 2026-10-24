import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ifixai.reporting.scorecard import grading_vendor

PACKAGED_FALLBACKS_PATH: Path = Path(__file__).parent / "judge_fallbacks.json"
PROJECT_FALLBACKS_FILENAME = "ifixai.judge-fallbacks.json"
FALLBACKS_PATH_ENV_VAR = "IFIXAI_JUDGE_FALLBACKS"

DEFAULT_ATTEMPTS_PER_MODEL = 3


class JudgeFallbackModel(BaseModel):
    """One candidate judge model in a provider's fallback chain."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    note: str = ""


class JudgeFallbackChain(BaseModel):
    """The ordered fallback models for a single judge provider."""

    model_config = ConfigDict(extra="forbid")

    attempts_per_model: int = Field(default=DEFAULT_ATTEMPTS_PER_MODEL, ge=1, le=10)
    models: list[JudgeFallbackModel] = Field(default_factory=list)


class JudgeFallbackPolicy(BaseModel):
    """The whole fallback file: one chain per judge provider name."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    providers: dict[str, JudgeFallbackChain] = Field(default_factory=dict)

    @field_validator("providers")
    @classmethod
    def _normalize_provider_keys(
        cls, value: dict[str, JudgeFallbackChain]
    ) -> dict[str, JudgeFallbackChain]:
        # chain_for looks providers up case-insensitively, so keys must match that.
        normalized: dict[str, JudgeFallbackChain] = {}
        for key, chain in value.items():
            name = key.strip().lower()
            if name in normalized:
                raise ValueError(f"duplicate provider key {key!r}")
            normalized[name] = chain
        return normalized

    def chain_for(self, provider: str) -> JudgeFallbackChain:
        """The chain declared for `provider`, or an empty one (no fallbacks)."""
        return self.providers.get(provider.strip().lower(), JudgeFallbackChain())


class JudgeFallbackConfigError(ValueError):
    """The user-supplied fallback JSON is missing, unreadable or malformed.

    Raised at judge construction — before any inspection runs — so a typo in the
    file surfaces immediately instead of half way through a paid run.
    """

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(
            f"Judge fallback config {path} is unusable: {reason}. "
            f"Fix the file, or unset {FALLBACKS_PATH_ENV_VAR} / remove "
            f"{PROJECT_FALLBACKS_FILENAME} to use the packaged defaults."
        )
        self.path = path
        self.reason = reason


def fallbacks_path(start_dir: Path | None = None) -> Path:
    """Resolve which fallback JSON to read: env override, project file, packaged."""
    override = os.environ.get(FALLBACKS_PATH_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    project = (start_dir or Path.cwd()) / PROJECT_FALLBACKS_FILENAME
    if project.is_file():
        return project
    return PACKAGED_FALLBACKS_PATH


def load_fallback_policy(path: Path | None = None) -> JudgeFallbackPolicy:
    """Read and validate the judge fallback JSON at `path` (auto-resolved if None)."""
    resolved = path or fallbacks_path()
    try:
        raw_text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise JudgeFallbackConfigError(resolved, f"cannot be read ({exc})") from exc
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise JudgeFallbackConfigError(resolved, f"is not valid JSON ({exc})") from exc
    if not isinstance(raw, dict):
        raise JudgeFallbackConfigError(
            resolved, "must hold a JSON object at the top level"
        )
    try:
        return JudgeFallbackPolicy(**raw)
    except ValidationError as exc:
        raise JudgeFallbackConfigError(resolved, f"has invalid fields\n{exc}") from exc


def resolve_judge_model_chain(
    provider: str,
    primary_model: str | None,
    policy: JudgeFallbackPolicy | None = None,
    excluded_model: str | None = None,
    excluded_provider: str | None = None,
) -> list[str | None]:
    """Ordered judge models to try: the configured one, then its declared fallbacks.

    The configured model always leads, even when it also appears further down the
    file, so declaring a fallback never silently reorders the operator's choice.
    ``None`` means "whatever the provider defaults to" and is preserved as-is.

    ``excluded_provider`` and ``excluded_model`` identify the system under test.
    Exclude every fallback from its vendor, using the scorecard's canonical
    vendor resolution so dated IDs, related models and aggregator aliases cannot
    silently turn an independent judge into a same-vendor judge. The configured
    primary remains the operator's explicit choice, including self-grading.
    """
    excluded_vendor = grading_vendor(excluded_provider, excluded_model) if excluded_provider else None
    chain = (policy or load_fallback_policy()).chain_for(provider)
    ordered: list[str | None] = [primary_model]
    seen: set[str | None] = {primary_model, excluded_model}
    for candidate in chain.models:
        if excluded_vendor is not None and grading_vendor(provider, candidate.model) == excluded_vendor:
            continue
        if candidate.model not in seen:
            ordered.append(candidate.model)
            seen.add(candidate.model)
    return ordered
