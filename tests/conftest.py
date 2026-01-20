"""Pytest configuration for unlockedpd tests."""
import pytest


@pytest.fixture(autouse=True)
def reset_unlockedpd():
    """Reset unlockedpd state before each test."""
    import unlockedpd
    # Ensure patches are applied
    unlockedpd.config.enabled = True
    yield
    # Reset after test
    unlockedpd.config.enabled = True
