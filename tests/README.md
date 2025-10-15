# DLHT Test Suite

This directory contains comprehensive test cases for the LEAD (LEArned Distributed Hash Table) implementation.

## Test Files

- **`test_models.py`**: Tests for RMI (Recursive Model Index) learned hash function
  - LinearModel tests
  - CubicModel tests
  - RecursiveModelIndex tests
  - Model training, prediction, serialization
  - Error bound calculation

- **`test_peer.py`**: Tests for LEADPeer (virtual node) implementation
  - Finger table operations
  - Range queries
  - Key storage and retrieval
  - Successor/predecessor management
  - Key transfer operations
  - Thread safety

- **`test_node.py`**: Tests for LEADNode (physical node) implementation
  - Node initialization and lifecycle
  - Put/get operations
  - Range queries
  - Model retraining
  - Federated model updates
  - Multi-node scenarios with bootstrap

- **`conftest.py`**: Shared pytest fixtures and configuration

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/test_models.py
pytest tests/test_peer.py
pytest tests/test_node.py
```

### Run Specific Test Class
```bash
pytest tests/test_models.py::TestLinearModel
pytest tests/test_peer.py::TestLEADPeer
```

### Run Specific Test
```bash
pytest tests/test_models.py::TestLinearModel::test_train_with_simple_data
```

### Run with Verbose Output
```bash
pytest -v
```

### Run with Coverage
```bash
pytest --cov=registry_service.dlht --cov-report=html
```

### Run Only Fast Tests (excluding slow tests)
```bash
pytest -m "not slow"
```

## Test Markers

Tests can be marked with custom markers:

- `@pytest.mark.slow`: Marks slow-running tests
- `@pytest.mark.integration`: Marks integration tests
- `@pytest.mark.unit`: Marks unit tests

## Test Coverage

The test suite covers:

1. **Model Training & Prediction**
   - Linear and cubic regression models
   - Two-stage RMI architecture
   - Error bound calculation
   - Model serialization

2. **DHT Operations**
   - Key routing via finger tables
   - Successor/predecessor management
   - Stabilization protocol
   - Fault tolerance with backup successors

3. **Data Storage**
   - Put/get operations
   - Range queries
   - Key transfer on node join
   - Concurrent access

4. **Model Updates**
   - Full model retraining
   - Federated model updates
   - Update threshold triggering

5. **Multi-Node Scenarios**
   - Node joining via bootstrap
   - Network formation
   - Data distribution

## Implementation Validation

These tests validate the Python implementation against the reference C implementation from the [LEAD repository](https://github.com/ShengzeWang/LEAD). Key validations include:

- ✅ RMI two-stage architecture matches
- ✅ Chord DHT routing protocol matches
- ✅ Range query traversal logic matches
- ✅ Model update threshold (40%) matches
- ✅ Error bounds are calculated and tracked
- ✅ Successor lists are maintained for fault tolerance
- ✅ SortedDict used for efficient range queries

## Known Issues

### Port Conflicts
Some tests use specific ports (5000-12000 range). If tests fail with "Address already in use" errors:
- Ensure no other processes are using these ports
- Wait a few seconds between test runs for ports to be released
- Run tests individually if conflicts persist

### Timing Issues
Some integration tests involve network operations and timeouts:
- Tests include `time.sleep()` calls to allow network initialization
- Increase sleep durations if tests fail intermittently
- Use `pytest --tb=short` to see abbreviated tracebacks

## Continuous Integration

To run tests in CI/CD:

```yaml
# Example GitHub Actions workflow
- name: Install dependencies
  run: |
    pip install -r requirements.txt

- name: Run tests
  run: |
    pytest tests/ -v --tb=short

- name: Run tests with coverage
  run: |
    pytest --cov=registry_service.dlht --cov-report=xml
```

## Adding New Tests

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Use fixtures from `conftest.py` for common setup
3. Add appropriate markers (`@pytest.mark.slow`, etc.)
4. Document complex test scenarios with comments
5. Ensure tests are deterministic (use `np.random.seed()` for randomness)

## Performance Benchmarks

For performance testing:

```bash
# Run with timing info
pytest --durations=10

# Profile specific tests
pytest tests/test_node.py::TestLEADNode::test_concurrent_operations --profile
```

## Debugging Tests

For debugging failed tests:

```bash
# Drop into pdb on failure
pytest --pdb

# Show local variables on failure
pytest -l

# Show full tracebacks
pytest --tb=long
```
