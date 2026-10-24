import json

import pytest

from ifixai.judge.fallbacks import (
    JudgeFallbackConfigError,
    load_fallback_policy,
    resolve_judge_model_chain,
)


def _write(tmp_path, providers):
    path = tmp_path / "fallbacks.json"
    path.write_text(json.dumps({"providers": providers}), encoding="utf-8")
    return path


@pytest.mark.parametrize("key", ["OpenRouter", " openrouter ", "OPENROUTER"])
def test_mixed_case_provider_key_is_honored(tmp_path, key):
    policy = load_fallback_policy(_write(tmp_path, {key: {"models": [{"model": "backup-model"}]}}))
    assert [m.model for m in policy.chain_for("openrouter").models] == ["backup-model"]
    assert resolve_judge_model_chain("OpenRouter", "primary-model", policy) == [
        "primary-model",
        "backup-model",
    ]


def test_provider_keys_differing_only_in_case_are_rejected(tmp_path):
    path = _write(tmp_path, {
        "openrouter": {"models": [{"model": "a"}]},
        "OpenRouter": {"models": [{"model": "b"}]},
    })
    with pytest.raises(JudgeFallbackConfigError, match="duplicate provider key"):
        load_fallback_policy(path)
