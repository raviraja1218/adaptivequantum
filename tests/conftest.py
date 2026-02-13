"""Pytest configuration for AdaptiveQuantum tests."""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture for test data directory."""
    return os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
