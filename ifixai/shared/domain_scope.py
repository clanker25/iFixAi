"""Whether a fixture names a specific domain for its agent to keep to (B32)."""

from ifixai.core.types import Fixture

# Domains that name no topic: a general-purpose agent answers anything, so off-topic refusal
# would itself be wrong.
GENERIC_DOMAINS: frozenset[str] = frozenset(
    {"", "generic", "general", "general purpose", "general-purpose", "all", "any", "n/a"}
)


def has_domain_restriction(fixture: Fixture) -> bool:
    """Whether `fixture.metadata.domain` names a specific scope. Only the domain counts: a system
    prompt doesn't make an agent domain-restricted."""
    domain = (fixture.metadata.domain or "").lower().strip()
    return domain not in GENERIC_DOMAINS
