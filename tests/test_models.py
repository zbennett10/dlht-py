"""
Tests for learned models (RMI) implementation
"""

import pytest
import numpy as np
from registry_service.dlht.models import LinearModel, CubicModel, RecursiveModelIndex


class TestLinearModel:
    """Test cases for LinearModel"""

    def test_initialization(self):
        """Test model initialization"""
        model = LinearModel()
        assert model.slope == 0.0
        assert model.intercept == 0.0
        assert model.max_error == 0.0

    def test_train_with_simple_data(self):
        """Test training with simple linear data"""
        model = LinearModel()
        keys = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        model.train(keys, positions)

        assert model.slope > 0
        assert model.intercept >= 0
        assert model.max_error >= 0

    def test_train_with_empty_data(self):
        """Test training with empty data"""
        model = LinearModel()
        keys = np.array([])
        positions = np.array([])

        model.train(keys, positions)
        # Should not crash

    def test_train_with_identical_keys(self):
        """Test training with identical keys"""
        model = LinearModel()
        keys = np.array([5.0, 5.0, 5.0])
        positions = np.array([0.0, 0.5, 1.0])

        model.train(keys, positions)

        assert model.slope == 0.0
        assert model.intercept == 0.5
        assert model.max_error == 0.0

    def test_predict(self):
        """Test prediction"""
        model = LinearModel()
        keys = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        model.train(keys, positions)

        # Test interpolation
        pred = model.predict(20.0)
        assert 0.0 <= pred <= 1.0

        # Test extrapolation (should be clamped)
        pred_low = model.predict(-10.0)
        pred_high = model.predict(50.0)
        assert 0.0 <= pred_low <= 1.0
        assert 0.0 <= pred_high <= 1.0

    def test_error_bounds(self):
        """Test that error bounds are calculated"""
        model = LinearModel()
        # Create data with some noise
        keys = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        positions = np.array([0.0, 0.3, 0.5, 0.7, 1.0])  # Not perfectly linear

        model.train(keys, positions)

        assert model.max_error > 0  # Should have some error
        assert model.max_error < 1.0  # But not too large

    def test_serialization(self):
        """Test model serialization and deserialization"""
        model = LinearModel()
        keys = np.array([0.0, 10.0, 20.0])
        positions = np.array([0.0, 0.5, 1.0])
        model.train(keys, positions)

        # Serialize
        params = model.to_dict()
        assert params['type'] == 'linear'
        assert 'slope' in params
        assert 'intercept' in params
        assert 'max_error' in params

        # Deserialize
        model2 = LinearModel.from_dict(params)
        assert model2.slope == model.slope
        assert model2.intercept == model.intercept
        assert model2.max_error == model.max_error


class TestCubicModel:
    """Test cases for CubicModel"""

    def test_initialization(self):
        """Test model initialization"""
        model = CubicModel()
        assert len(model.coeffs) == 4
        assert model.max_error == 0.0

    def test_train_with_sufficient_data(self):
        """Test training with sufficient data points"""
        model = CubicModel()
        keys = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        # Cubic data: y = 0.01*x^3
        positions = np.array([0.0, 0.01, 0.08, 0.27, 0.64, 1.25])

        model.train(keys, positions)

        assert model.coeffs is not None
        assert model.max_error >= 0

    def test_train_with_insufficient_data(self):
        """Test training with insufficient data points (< 4)"""
        model = CubicModel()
        keys = np.array([0.0, 1.0, 2.0])
        positions = np.array([0.0, 0.5, 1.0])

        model.train(keys, positions)

        # Should use default coefficients
        assert np.array_equal(model.coeffs, np.array([0, 0, 0, 0.5]))

    def test_predict(self):
        """Test prediction"""
        model = CubicModel()
        keys = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        positions = np.array([0.0, 0.1, 0.3, 0.6, 0.9, 1.0])

        model.train(keys, positions)

        pred = model.predict(25.0)
        assert 0.0 <= pred <= 1.0

    def test_serialization(self):
        """Test model serialization and deserialization"""
        model = CubicModel()
        keys = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        positions = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        model.train(keys, positions)

        # Serialize
        params = model.to_dict()
        assert params['type'] == 'cubic'
        assert 'coeffs' in params
        assert 'max_error' in params

        # Deserialize
        model2 = CubicModel.from_dict(params)
        assert np.array_equal(model2.coeffs, model.coeffs)
        assert model2.max_error == model.max_error


class TestRecursiveModelIndex:
    """Test cases for RecursiveModelIndex (RMI)"""

    def test_initialization_linear(self):
        """Test RMI initialization with linear models"""
        rmi = RecursiveModelIndex(
            branching_factor=10,
            model_type='linear',
            stage1_model_type='linear'
        )

        assert rmi.branching_factor == 10
        assert len(rmi.stage2_models) == 10
        assert isinstance(rmi.stage1_model, LinearModel)
        assert all(isinstance(m, LinearModel) for m in rmi.stage2_models)

    def test_initialization_cubic(self):
        """Test RMI initialization with cubic stage1"""
        rmi = RecursiveModelIndex(
            branching_factor=10,
            model_type='linear',
            stage1_model_type='cubic'
        )

        assert isinstance(rmi.stage1_model, CubicModel)

    def test_train_with_sequential_keys(self):
        """Test training with sequential keys"""
        rmi = RecursiveModelIndex(branching_factor=10, model_type='linear')

        # Create sequential keys
        keys = np.arange(0, 1000, dtype=float)
        rmi.train(keys)

        assert rmi.version == 1

        # Test predictions are in valid range
        for key in [0, 100, 500, 999]:
            hash_value = rmi.predict(key)
            assert 0 <= hash_value < rmi.hash_space_size

    def test_train_with_random_keys(self):
        """Test training with random keys"""
        rmi = RecursiveModelIndex(branching_factor=20, model_type='linear')

        # Create random keys
        np.random.seed(42)
        keys = np.random.randint(0, 1000000, size=5000)
        rmi.train(keys)

        assert rmi.version == 1

    def test_predict_maintains_order(self):
        """Test that learned hash approximately maintains key order"""
        rmi = RecursiveModelIndex(branching_factor=50, model_type='linear')

        # Train on sequential keys
        keys = np.arange(0, 10000, 10, dtype=float)
        rmi.train(keys)

        # Check that predictions are monotonically increasing (approximately)
        predictions = [rmi.predict(k) for k in [100, 200, 300, 400, 500]]

        # Most predictions should be in order (allowing for some model error)
        ordered_count = sum(1 for i in range(len(predictions) - 1)
                          if predictions[i] <= predictions[i + 1])
        assert ordered_count >= len(predictions) - 2  # Allow 1-2 inversions

    def test_predict_with_error(self):
        """Test prediction with error bounds"""
        rmi = RecursiveModelIndex(branching_factor=10, model_type='linear')

        keys = np.arange(0, 1000, dtype=float)
        rmi.train(keys)

        hash_value, error = rmi.predict_with_error(500.0)

        assert 0 <= hash_value < rmi.hash_space_size
        assert error >= 0
        assert error < rmi.hash_space_size  # Error should be reasonable

    def test_update_leaf(self):
        """Test updating a specific leaf model"""
        rmi = RecursiveModelIndex(branching_factor=10, model_type='linear')

        # Train initial model
        keys = np.arange(0, 1000, dtype=float)
        rmi.train(keys)

        initial_version = rmi.version

        # Update a specific leaf
        bucket_id = 5
        new_keys = np.array([500.0, 510.0, 520.0])
        new_positions = np.array([0.5, 0.51, 0.52])

        rmi.update_leaf(bucket_id, new_keys, new_positions)

        # Version should not change for leaf-only updates
        assert rmi.version == initial_version

    def test_serialization(self):
        """Test RMI serialization and deserialization"""
        rmi = RecursiveModelIndex(
            branching_factor=20,
            model_type='linear',
            stage1_model_type='cubic'
        )

        keys = np.arange(0, 1000, dtype=float)
        rmi.train(keys)

        # Serialize
        data = rmi.to_dict()
        assert data['branching_factor'] == 20
        assert data['model_type'] == 'linear'
        assert data['stage1_model_type'] == 'cubic'
        assert data['version'] == 1
        assert len(data['stage2_models']) == 20

        # Deserialize
        rmi2 = RecursiveModelIndex.from_dict(data)
        assert rmi2.branching_factor == rmi.branching_factor
        assert rmi2.version == rmi.version
        assert len(rmi2.stage2_models) == len(rmi.stage2_models)

        # Predictions should match
        test_key = 500.0
        assert rmi2.predict(test_key) == rmi.predict(test_key)

    def test_hash_distribution(self):
        """Test that hash values are well-distributed"""
        rmi = RecursiveModelIndex(branching_factor=100, model_type='linear')

        keys = np.arange(0, 10000, dtype=float)
        rmi.train(keys)

        # Sample predictions
        sample_size = 1000
        predictions = [rmi.predict(float(k)) for k in range(0, 10000, 10)]

        # Check distribution (should cover significant portion of hash space)
        min_pred = min(predictions)
        max_pred = max(predictions)
        coverage = (max_pred - min_pred) / rmi.hash_space_size

        # Should cover at least some portion of the hash space
        assert coverage > 0.0

    def test_empty_train(self):
        """Test training with empty dataset"""
        rmi = RecursiveModelIndex(branching_factor=10, model_type='linear')

        keys = np.array([])
        rmi.train(keys)  # Should not crash

        # Predictions should still work (using default models)
        hash_value = rmi.predict(100.0)
        assert 0 <= hash_value < rmi.hash_space_size


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
