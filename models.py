"""
Learned models for LEAD DHT (Recursive Model Index implementation)
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class LinearModel:
    """Simple linear regression model for RMI leaf nodes"""
    
    def __init__(self):
        self.slope = 0.0
        self.intercept = 0.0
        self.offset = 0.0  # Anchor offset for dynamic adjustment
        self.scale = 1.0   # Anchor scale for dynamic adjustment
        self.min_key = 0.0
        self.max_key = 1.0
        
    def train(self, keys: np.ndarray, positions: np.ndarray):
        """Train linear model on keys and their positions"""
        if len(keys) == 0:
            return
        
        # Store normalization parameters
        self.min_key = np.min(keys)
        self.max_key = np.max(keys)
        
        if self.max_key == self.min_key:
            self.slope = 0.0
            self.intercept = 0.5
            return
            
        normalized_keys = (keys - self.min_key) / (self.max_key - self.min_key)
        
        # Linear regression
        n = len(keys)
        sum_x = np.sum(normalized_keys)
        sum_y = np.sum(positions)
        sum_xy = np.sum(normalized_keys * positions)
        sum_x2 = np.sum(normalized_keys ** 2)
        
        denominator = n * sum_x2 - sum_x ** 2
        if abs(denominator) < 1e-10:
            self.slope = 0.0
            self.intercept = 0.5
        else:
            self.slope = (n * sum_xy - sum_x * sum_y) / denominator
            self.intercept = (sum_y - self.slope * sum_x) / n
        
    def predict(self, key: float) -> float:
        """Predict position for a key"""
        # Normalize key
        if self.max_key == self.min_key:
            normalized = 0.5
        else:
            normalized = (key - self.min_key) / (self.max_key - self.min_key)
            normalized = np.clip(normalized, 0, 1)
        
        prediction = self.slope * normalized + self.intercept
        # Apply anchor adjustments
        return self.offset + self.scale * prediction
        
    def update_anchor(self, keys: np.ndarray):
        """Update anchor parameters based on key distribution"""
        if len(keys) == 0:
            return
            
        predictions = np.array([self.predict(k) for k in keys])
        
        # Adjust offset to center median
        median_pred = np.median(predictions)
        self.offset -= (median_pred - 0.5)
        
        # Adjust scale so 95th percentile is near 1.0
        p95 = np.percentile(predictions, 95)
        if p95 > 0.1:  # Avoid division by very small numbers
            self.scale *= 0.95 / p95
            
    def to_dict(self) -> dict:
        """Serialize model parameters"""
        return {
            'type': 'linear',
            'slope': float(self.slope),
            'intercept': float(self.intercept),
            'offset': float(self.offset),
            'scale': float(self.scale),
            'min_key': float(self.min_key),
            'max_key': float(self.max_key)
        }
        
    @classmethod
    def from_dict(cls, params: dict) -> 'LinearModel':
        """Deserialize model parameters"""
        model = cls()
        model.slope = params['slope']
        model.intercept = params['intercept']
        model.offset = params['offset']
        model.scale = params['scale']
        model.min_key = params['min_key']
        model.max_key = params['max_key']
        return model


class CubicModel:
    """Cubic polynomial model for RMI leaf nodes"""
    
    def __init__(self):
        self.coeffs = np.zeros(4)  # a*x^3 + b*x^2 + c*x + d
        self.offset = 0.0
        self.scale = 1.0
        self.min_key = 0.0
        self.max_key = 1.0
        
    def train(self, keys: np.ndarray, positions: np.ndarray):
        """Train cubic model using polynomial fitting"""
        if len(keys) < 4:  # Need at least 4 points for cubic
            self.coeffs = np.array([0, 0, 0, 0.5])
            return
        
        self.min_key = np.min(keys)
        self.max_key = np.max(keys)
        
        if self.max_key == self.min_key:
            self.coeffs = np.array([0, 0, 0, 0.5])
            return
            
        normalized_keys = (keys - self.min_key) / (self.max_key - self.min_key)
        self.coeffs = np.polyfit(normalized_keys, positions, 3)
        
    def predict(self, key: float) -> float:
        """Predict position for a key"""
        if self.max_key == self.min_key:
            normalized = 0.5
        else:
            normalized = (key - self.min_key) / (self.max_key - self.min_key)
            normalized = np.clip(normalized, 0, 1)
        
        prediction = np.polyval(self.coeffs, normalized)
        return self.offset + self.scale * prediction
        
    def update_anchor(self, keys: np.ndarray):
        """Update anchor parameters"""
        if len(keys) == 0:
            return
            
        predictions = np.array([self.predict(k) for k in keys])
        median_pred = np.median(predictions)
        self.offset -= (median_pred - 0.5)
        
        p95 = np.percentile(predictions, 95)
        if p95 > 0.1:
            self.scale *= 0.95 / p95
            
    def to_dict(self) -> dict:
        """Serialize model parameters"""
        return {
            'type': 'cubic',
            'coeffs': self.coeffs.tolist(),
            'offset': float(self.offset),
            'scale': float(self.scale),
            'min_key': float(self.min_key),
            'max_key': float(self.max_key)
        }
        
    @classmethod
    def from_dict(cls, params: dict) -> 'CubicModel':
        """Deserialize model parameters"""
        model = cls()
        model.coeffs = np.array(params['coeffs'])
        model.offset = params['offset']
        model.scale = params['scale']
        model.min_key = params['min_key']
        model.max_key = params['max_key']
        return model


class RecursiveModelIndex:
    """Two-stage Recursive Model Index for learned hashing"""
    
    def __init__(self, branching_factor: int = 100, model_type: str = 'linear',
                 hash_space_size: int = 2**160):
        self.branching_factor = branching_factor
        self.model_type = model_type
        self.hash_space_size = hash_space_size
        self.stage1_model = LinearModel()  # Root model is always linear
        self.stage2_models = []  # Leaf models
        self.version = 0
        
        # Initialize leaf models
        for _ in range(branching_factor):
            if model_type == 'cubic':
                self.stage2_models.append(CubicModel())
            else:
                self.stage2_models.append(LinearModel())
                
    def train(self, keys: np.ndarray):
        """Train the RMI on a dataset of keys"""
        if len(keys) == 0:
            logger.warning("Training RMI with empty dataset")
            return
            
        # Sort keys
        sorted_keys = np.sort(keys)
        n = len(sorted_keys)
        
        # Normalize keys to [0, 1]
        min_key = np.min(sorted_keys)
        max_key = np.max(sorted_keys)
        
        if max_key == min_key:
            logger.warning("All keys are identical, using default model")
            return
            
        normalized_keys = (sorted_keys - min_key) / (max_key - min_key)
        positions = np.arange(n) / n
        
        # Train stage 1 model to predict bucket
        bucket_ids = np.floor(positions * self.branching_factor).astype(int)
        bucket_ids = np.clip(bucket_ids, 0, self.branching_factor - 1)
        self.stage1_model.train(normalized_keys, bucket_ids)
        
        # Train stage 2 models
        for bucket_id in range(self.branching_factor):
            mask = bucket_ids == bucket_id
            if np.sum(mask) > 0:
                bucket_keys = normalized_keys[mask]
                bucket_positions = (positions[mask] - bucket_id / self.branching_factor) * self.branching_factor
                self.stage2_models[bucket_id].train(bucket_keys, bucket_positions)
                
        self.version += 1
        logger.info(f"RMI trained on {n} keys, version {self.version}")
        
    def predict(self, key: float) -> int:
        """Predict hash value for a key"""
        # Normalize key
        normalized_key = key / self.hash_space_size
        
        # Stage 1: predict bucket
        bucket_prediction = self.stage1_model.predict(normalized_key)
        bucket_id = int(np.clip(bucket_prediction, 0, self.branching_factor - 1))
        
        # Stage 2: predict position within bucket
        position_in_bucket = self.stage2_models[bucket_id].predict(normalized_key)
        
        # Convert to hash value
        overall_position = (bucket_id + np.clip(position_in_bucket, 0, 1)) / self.branching_factor
        hash_value = int(overall_position * self.hash_space_size)
        
        return max(0, min(hash_value, self.hash_space_size - 1))
        
    def update_leaf(self, bucket_id: int, keys: np.ndarray, positions: np.ndarray):
        """Update a specific leaf model (for federated updates)"""
        if 0 <= bucket_id < self.branching_factor and len(keys) > 0:
            self.stage2_models[bucket_id].train(keys, positions)
            
    def get_leaf_params(self, bucket_id: int) -> dict:
        """Get parameters of a leaf model for federated updates"""
        if 0 <= bucket_id < self.branching_factor:
            return self.stage2_models[bucket_id].to_dict()
        return {}
        
    def set_leaf_params(self, bucket_id: int, params: dict):
        """Set parameters of a leaf model"""
        if 0 <= bucket_id < self.branching_factor:
            if params['type'] == 'linear':
                self.stage2_models[bucket_id] = LinearModel.from_dict(params)
            elif params['type'] == 'cubic':
                self.stage2_models[bucket_id] = CubicModel.from_dict(params)
                
    def to_dict(self) -> dict:
        """Serialize entire model"""
        return {
            'version': self.version,
            'branching_factor': self.branching_factor,
            'model_type': self.model_type,
            'hash_space_size': self.hash_space_size,
            'stage1_model': self.stage1_model.to_dict(),
            'stage2_models': [m.to_dict() for m in self.stage2_models]
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'RecursiveModelIndex':
        """Deserialize model"""
        rmi = cls(
            branching_factor=data['branching_factor'],
            model_type=data['model_type'],
            hash_space_size=data['hash_space_size']
        )
        rmi.version = data['version']
        rmi.stage1_model = LinearModel.from_dict(data['stage1_model'])
        
        rmi.stage2_models = []
        for params in data['stage2_models']:
            if params['type'] == 'linear':
                rmi.stage2_models.append(LinearModel.from_dict(params))
            elif params['type'] == 'cubic':
                rmi.stage2_models.append(CubicModel.from_dict(params))
                
        return rmi