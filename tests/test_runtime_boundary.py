"""The default engine must run with optional reference/experiment packages absent."""

import json
import os
import subprocess
import sys
from textwrap import dedent


def test_default_cli_runs_without_reference_or_experiments():
    program = dedent("""
        import importlib.abc
        import runpy
        import sys

        class RuntimeOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'reference', 'experiments'}:
                    raise ImportError('Default runtime imported optional package: ' + fullname)

        sys.meta_path.insert(0, RuntimeOnly())
        sys.argv = ['mini-hisparse', 'demo', '--device', 'cpu', '--top-k', '2',
                    '--prompt-tokens', '65', '--gpu-pages', '12', '--requests', '1',
                    '--max-new-tokens', '3']
        runpy.run_module('mini_hisparse', run_name='__main__')
    """)
    result = subprocess.run(
        [sys.executable, "-c", program],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["cache_layout"] == "hybrid_rows"
    assert report["requests"]["0"] == {
        "prefill_count": 1,
        "decode_count": 2,
        "generated_tokens": 3,
    }
