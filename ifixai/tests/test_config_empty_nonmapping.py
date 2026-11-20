from pathlib import Path

import pytest

from ifixai.cli.config_file import load_config


@pytest.mark.parametrize('document', ['[]', 'false', '0', '""'])
def test_falsey_nonmapping_config_is_rejected(tmp_path: Path, document: str) -> None:
    (tmp_path / 'ifixai.yaml').write_text(document, encoding='utf-8')
    with pytest.raises(ValueError, match='mapping'):
        load_config(tmp_path)


@pytest.mark.parametrize('document', ['', '# empty config', '{}', 'null'])
def test_empty_config_remains_supported(tmp_path: Path, document: str) -> None:
    (tmp_path / 'ifixai.yaml').write_text(document, encoding='utf-8')
    assert load_config(tmp_path) is not None


@pytest.mark.parametrize('key', ['1', 'true', 'null'])
def test_nonstring_config_keys_are_rejected(tmp_path: Path, key: str) -> None:
    (tmp_path / 'ifixai.yaml').write_text(
        f'provider: mock\n{key}: unexpected\n', encoding='utf-8'
    )
    with pytest.raises(ValueError, match='string keys at the top level'):
        load_config(tmp_path)
