"""
=============================================================================
Unit Tests for Landslide Detection Model
=============================================================================
"""

import pytest
import numpy as np
import pandas as pd
from landslide_detection_model import (
    LandslideDataSimulator,
    LandslideFeatureEngineer,
    LandslideDetectionModel
)


class TestDataSimulator:
    """Test data generation and synthetic dataset quality"""
    
    def test_simulator_initialization(self):
        """Test simulator can be instantiated"""
        sim = LandslideDataSimulator(n_samples=100)
        assert sim.n_samples == 100
    
    def test_data_generation(self):
        """Test dataset generation produces correct shape"""
        sim = LandslideDataSimulator(n_samples=500)
        df, labels = sim.generate()
        
        assert df.shape[0] == 500
        assert df.shape[1] == 12  # 12 sensor parameters
        assert len(labels) == 500
        assert labels.dtype == int
    
    def test_label_distribution(self):
        """Test label class distribution is reasonable"""
        sim = LandslideDataSimulator(n_samples=5000)
        df, labels = sim.generate()
        
        # Should have all 3 classes
        unique_labels = np.unique(labels)
        assert len(unique_labels) == 3
        assert 0 in unique_labels and 2 in unique_labels
        
        # Class distribution should be roughly: 50% low, 30% moderate, 20% high
        class_counts = np.bincount(labels)
        assert class_counts[0] > class_counts[1] > class_counts[2]
    
    def test_sensor_ranges(self):
        """Test sensor values are within expected physical ranges"""
        sim = LandslideDataSimulator(n_samples=1000)
        df, _ = sim.generate()
        
        assert (df['rainfall_intensity_mm_hr'] >= 0).all()
        assert (df['soil_moisture_percentage'] >= 5).all() and (df['soil_moisture_percentage'] <= 98).all()
        assert (df['slope_angle_deg'] >= 10).all() and (df['slope_angle_deg'] <= 65).all()


class TestFeatureEngineer:
    """Test feature engineering pipeline"""
    
    def test_engineer_initialization(self):
        """Test feature engineer can be instantiated"""
        engineer = LandslideFeatureEngineer()
        assert engineer.feature_names is None
    
    def test_feature_transformation(self):
        """Test feature engineering produces expected output"""
        sim = LandslideDataSimulator(n_samples=100)
        df, _ = sim.generate()
        
        engineer = LandslideFeatureEngineer()
        X_transformed = engineer.transform(df)
        
        # Should have 12 original + 6 engineered = 18 total
        assert X_transformed.shape[1] == 18
        assert X_transformed.shape[0] == 100
    
    def test_feature_names_consistency(self):
        """Test feature names are consistent across calls"""
        sim = LandslideDataSimulator(n_samples=50)
        df1, _ = sim.generate()
        df2, _ = sim.generate()
        
        engineer = LandslideFeatureEngineer()
        X1 = engineer.transform(df1)
        feature_names_1 = engineer.feature_names.copy()
        
        X2 = engineer.transform(df2)
        feature_names_2 = engineer.feature_names
        
        assert feature_names_1 == feature_names_2


class TestLandslideDetectionModel:
    """Test the main ML model"""
    
    def test_model_initialization(self):
        """Test model can be instantiated"""
        model = LandslideDetectionModel()
        assert model.is_trained == False
        assert model.model is None
    
    def test_model_training(self):
        """Test model can be trained on synthetic data"""
        sim = LandslideDataSimulator(n_samples=1000)
        df, labels = sim.generate()
        
        model = LandslideDetectionModel()
        metrics = model.train(df, labels, test_size=0.2)
        
        assert model.is_trained == True
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert metrics['accuracy'] > 0.85  # Should achieve >85% accuracy
    
    def test_single_prediction(self):
        """Test single sample prediction"""
        sim = LandslideDataSimulator(n_samples=500)
        df, labels = sim.generate()
        
        model = LandslideDetectionModel()
        model.train(df, labels)
        
        # Test on a high-risk scenario
        sample = {
            'rainfall_intensity_mm_hr': 50.0,
            'cumulative_rainfall_72h_mm': 200.0,
            'soil_moisture_percentage': 85.0,
            'slope_angle_deg': 50.0,
            'slope_displacement_mm': 6.0,
            'pore_water_pressure_kpa': 40.0,
            'seismic_vibration_g': 0.1,
            'elevation_meters': 1500.0,
            'vegetation_ndvi': 0.3,
            'soil_shear_strength_kpa': 25.0,
            'fault_distance_km': 2.0,
            'groundwater_depth_m': 3.0
        }
        
        result = model.predict(sample)
        
        assert 'risk_level' in result
        assert 'risk_code' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert 'primary_triggers' in result
        assert 'recommended_action' in result
        
        assert result['risk_code'] in [0, 1, 2]
        assert 0 <= result['confidence'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
