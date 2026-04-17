import time
from headless_smoke import run_headless_smoke


def test_headless_smoke_quick():
    """Quick CI-friendly headless smoke test."""
    # short run to ensure no exceptions and thread exits
    assert run_headless_smoke(duration_s=0.5, steps=20) is True
