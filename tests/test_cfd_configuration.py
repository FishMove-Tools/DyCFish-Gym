"""CFD configuration checks without ROM imports or solver startup."""
import importlib
import sys
import pytest
from CFD_stage import training as cfd


def test_configuration_import_does_not_load_rom_or_start_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loaded_before = set(sys.modules)
    importlib.reload(cfd)
    assert not any(name.startswith('dynamic_stage') for name in set(sys.modules) - loaded_before)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('arguments', [[], ['--target', '1', '0'],
    ['--target', '1', '0', '--initializer', 'missing_separator'],
    ['--target', '1', '0', '--initializer', 'does_not_exist:initialize']])
def test_invalid_configuration_does_not_launch_workers(arguments, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid setup must fail before creating workers')
    monkeypatch.setattr(cfd.mp, 'Process', forbidden)
    with pytest.raises(SystemExit) as error:
        cfd.main(arguments)
    assert error.value.code == 2
