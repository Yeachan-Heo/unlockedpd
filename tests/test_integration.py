"""Integration tests for unlockedpd."""
import subprocess
import sys


class TestImportPatching:
    """Tests for import-time patching."""

    def test_import_patches_automatically(self):
        """Verify that importing unlockedpd patches pandas methods."""
        code = '''
import pandas as pd
original_mean = pd.core.window.rolling.Rolling.mean

import unlockedpd

patched_mean = pd.core.window.rolling.Rolling.mean
assert original_mean is not patched_mean, "Method should be patched after import"

from unlockedpd._patch import is_patched
assert is_patched(pd.core.window.rolling.Rolling, 'mean')
print("PASS")
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        assert result.returncode == 0, f"Test failed: {result.stderr}"
        assert "PASS" in result.stdout

    def test_import_warmup_is_lazy_by_default(self):
        """Verify default import patches pandas without loading warmup module."""
        code = '''
import sys
import pandas as pd
original_mean = pd.core.window.rolling.Rolling.mean

import unlockedpd

patched_mean = pd.core.window.rolling.Rolling.mean
assert original_mean is not patched_mean, "Method should be patched after import"
assert unlockedpd.config.warmup == "lazy"
assert "unlockedpd._warmup" not in sys.modules
print("PASS")
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        assert result.returncode == 0, f"Test failed: {result.stderr}"
        assert "PASS" in result.stdout

    def test_eager_import_policy_invokes_warmup(self):
        """Verify eager import policy invokes the warmup entrypoint."""
        code = '''
import os
import sys
import types

os.environ["UNLOCKEDPD_WARMUP"] = "eager"
module = types.ModuleType("unlockedpd._warmup")
module.called = False

def warmup_all():
    module.called = True

module.warmup_all = warmup_all
sys.modules["unlockedpd._warmup"] = module

import unlockedpd

assert unlockedpd.config.warmup == "eager"
assert module.called is True
print("PASS")
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        assert result.returncode == 0, f"Test failed: {result.stderr}"
        assert "PASS" in result.stdout

    def test_unpatch_restores_original(self):
        """Verify unpatch_all restores original methods."""
        code = '''
import pandas as pd
original_mean = pd.core.window.rolling.Rolling.mean

import unlockedpd
unlockedpd.unpatch_all()

restored_mean = pd.core.window.rolling.Rolling.mean
assert original_mean is restored_mean, "Method should be restored after unpatch_all"
print("PASS")
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        assert result.returncode == 0, f"Test failed: {result.stderr}"
        assert "PASS" in result.stdout


class TestConfigDisable:
    """Tests for config.enabled=False."""

    def test_config_enabled_false_uses_original(self):
        """Verify config.enabled=False bypasses optimization."""
        code = '''
import pandas as pd
import numpy as np
import unlockedpd

df = pd.DataFrame(np.random.randn(10, 3))

unlockedpd.config.enabled = True
result1 = df.rolling(3).mean()

unlockedpd.config.enabled = False
result2 = df.rolling(3).mean()

pd.testing.assert_frame_equal(result1, result2)
print("PASS")
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        assert result.returncode == 0, f"Test failed: {result.stderr}"
        assert "PASS" in result.stdout


class TestFallback:
    """Tests for graceful fallback."""

    def test_series_falls_back(self):
        """Verify Series operations fall back to pandas."""
        code = '''
import pandas as pd
import numpy as np
import unlockedpd

s = pd.Series(np.random.randn(100))

# Should fall back gracefully for Series
result = s.rolling(5).mean()
assert len(result) == 100
print("PASS")
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        assert result.returncode == 0, f"Test failed: {result.stderr}"
        assert "PASS" in result.stdout
