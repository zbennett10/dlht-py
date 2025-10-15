"""
Pytest configuration and fixtures for DLHT tests
"""

import pytest


@pytest.fixture(scope="session")
def sample_keys():
    """Fixture providing sample keys for testing"""
    import numpy as np
    return np.arange(0, 1000, dtype=float)


@pytest.fixture(scope="session")
def random_keys():
    """Fixture providing random keys for testing"""
    import numpy as np
    np.random.seed(42)
    return np.random.randint(0, 1000000, size=1000)


@pytest.fixture
def test_config():
    """Fixture providing test configuration"""
    from registry_service.dlht.config import LEADConfig
    return LEADConfig(
        ip='127.0.0.1',
        base_port=15000,
        num_virtual_nodes=3,
        stabilize_interval=0.5,
        rpc_timeout=2.0
    )
