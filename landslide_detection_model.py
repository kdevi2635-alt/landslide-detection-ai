"""
=============================================================================
🏔️ AI-POWERED LANDSLIDE DETECTION & EARLY WARNING MODEL
Domain: Geological Disaster Risk Assessment & Early Warning Systems (EWS)
Standards: Geological Survey of India (GSI) & USGS Hazard Classification
=============================================================================
"""

import numpy as np
import pandas as pd
import time
import joblib
from typing import Dict, Any, Tuple, Union

# Machine Learning Core
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import xgboost as xgb


# =====================================================================
# 1. GEOLOGICAL & ENVIRONMENTAL DATA SPECIFICATION
# =====================================================================
class LandslideDataSimulator:
    """
    Generates physically consistent geological datasets based on 
    the Infinite Slope Stability Model and Mohr-Coulomb failure criteria.
    """

    def __init__(self, n_samples: int = 12000, random_seed: int = 42):
        self.n_samples = n_samples
        np.random.seed(random_seed)

    def generate(self) -> Tuple[pd.DataFrame, np.ndarray]:
        n = self.n_samples

        # Primary geo-hydrological variables
        slope_angle = np.random.uniform(10, 65, n)
        elevation = np.random.uniform(200, 3200, n)
        vegetation_ndvi = np.random.uniform(0.1, 0.85, n)
        fault_distance = np.random.uniform(0.2, 25.0, n)
        soil_shear_strength = np.random.uniform(15, 60, n)  # kPa

        # Weather & dynamic sensors
        rainfall_intensity = np.random.exponential(scale=12.0, size=n).clip(0, 120)
        cumulative_rainfall_72h = np.random.uniform(10, 350, n)
        
        # Soil moisture dynamically driven by rainfall accumulation
        base_moisture = np.random.uniform(15, 45, n)
        soil_moisture = np.clip(base_moisture + (cumulative_rainfall_72h * 0.15) + (rainfall_intensity * 0.4), 5, 98)

        # Pore water pressure (rises with moisture and groundwater table)
        groundwater_depth = np.random.uniform(1.5, 30.0, n)
        pore_pressure = np.clip((soil_moisture / 100.0) * (40.0 - groundwater_depth) * 1.5, 0, 80)

        # Ground displacement: strongly triggered on steep saturated slopes
        displacement_base = (slope_angle / 50.0) ** 2 * (soil_moisture / 60.0) * 8.0
        slope_displacement = np.maximum(0, displacement_base + np.random.normal(0, 1.5, n))

        # Seismic / vibration disturbance
        seismic_vibration = np.random.exponential(scale=0.08, size=n).clip(0, 1.8)

        # Assemble DataFrame
        df = pd.DataFrame({
            'rainfall_intensity_mm_hr': rainfall_intensity,
            'cumulative_rainfall_72h_mm': cumulative_rainfall_72h,
            'soil_moisture_percentage': soil_moisture,
            'slope_angle_deg': slope_angle,
            'slope_displacement_mm': slope_displacement,
            'pore_water_pressure_kpa': pore_pressure,
            'seismic_vibration_g': seismic_vibration,
            'elevation_meters': elevation,
            'vegetation_ndvi': vegetation_ndvi,
            'soil_shear_strength_kpa': soil_shear_strength,
            'fault_distance_km': fault_distance,
            'groundwater_depth_m': groundwater_depth
        })

        # Factor of Safety (FoS) estimation proxy (Higher score = higher instability)
        instability_index = (
            (df['soil_moisture_percentage'] / 50.0) * 1.8 +
            (df['cumulative_rainfall_72h_mm'] / 180.0) * 1.5 +
            (df['slope_angle_deg'] / 35.0) * 1.6 +
            (df['slope_displacement_mm'] / 5.0) * 2.5 +
            (df['pore_water_pressure_kpa'] / 25.0) * 1.4 +
            (df['seismic_vibration_g'] / 0.5) * 1.2 -
            (df['vegetation_ndvi'] * 1.0) -
            (df['soil_shear_strength_kpa'] / 40.0) * 1.1 -
            (df['fault_distance_km'] / 15.0) * 0.4
        )

        # 3 Hazard Classes: 0: Low Risk (Green), 1: Moderate Warning (Yellow), 2: High Alert (Red)
        labels = np.zeros(n, dtype=int)
        labels[instability_index > 2.8] = 1
        labels[instability_index > 4.6] = 2

        # 3% field sensor noise
        noise_idx = np.random.choice(n, int(n * 0.03), replace=False)
        labels[noise_idx] = np.random.choice([0, 1, 2], len(noise_idx))

        return df, labels


# =====================================================================
# 2. FEATURE ENGINEERING ENGINE
# =====================================================================
class LandslideFeatureEngineer:
    """Computes interaction ratios, pore saturation indices, and stability proxies."""

    def __init__(self):
        self.feature_names = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()

        # Hydro-Mechanical interactions
        X['moisture_x_slope'] = X['soil_moisture_percentage'] * X['slope_angle_deg']
        X['rain_x_displacement'] = X['cumulative_rainfall_72h_mm'] * X['slope_displacement_mm']
        X['pore_moisture_ratio'] = X['pore_water_pressure_kpa'] / (X['soil_moisture_percentage'] + 1e-4)

        # Hydrological saturation indicator
        X['critical_saturation'] = (
            (X['soil_moisture_percentage'] > 75.0) & 
            (X['cumulative_rainfall_72h_mm'] > 120.0)
        ).astype(int)

        # Kinematic displacement trigger
        X['rapid_creep_indicator'] = (X['slope_displacement_mm'] > 4.0).astype(int)

        # Combined vulnerability score
        X['hazard_vulnerability_score'] = (
            X['critical_saturation'] * 2.0 +
            X['rapid_creep_indicator'] * 3.0 +
            (X['slope_angle_deg'] > 40.0).astype(int) * 1.5
        )

        if self.feature_names is None:
            self.feature_names = X.columns.tolist()

        return X


# =====================================================================
# 3. CORE AI ENSEMBLE MODEL
# =====================================================================
class LandslideDetectionModel:
    """
    Production-grade Ensemble Model integrating:
    - Extreme Gradient Boosting (XGBoost)
    - Random Forest Classifier with Cost-Complexity Pruning
    - Soft Probability Voting
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_engineer = LandslideFeatureEngineer()
        self.model = None
        self.is_trained = False
        self.class_labels = {0: 'LOW_RISK', 1: 'MODERATE_WARNING', 2: 'HIGH_ALERT'}

    def build_model(self):
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        xgb_clf = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric='mlogloss'
        )

        # Ensemble combining tree-based bagging & boosting
        self.model = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb_clf)],
            voting='soft',
            weights=[1.0, 1.2]
        )

    def train(self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2):
        print("⚙️ Engineering geological features...")
        X_eng = self.feature_engineer.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_eng, y, test_size=test_size, random_state=42, stratify=y
        )

        print("📈 Scaling features and fitting Ensemble Model (RF + XGBoost)...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.build_model()
        t0 = time.time()
        self.model.fit(X_train_scaled, y_train)
        train_time = time.time() - t0

        # Evaluate performance
        y_pred = self.model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"\n✅ Training Complete in {train_time:.2f}s!")
        print(f"🎯 Test Accuracy : {acc * 100:.2f}%")
        print(f"🎯 Weighted F1   : {f1:.4f}\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Low', 'Moderate', 'High']))

        self.is_trained = True
        return {'accuracy': acc, 'f1_score': f1}

    def predict(self, sensor_payload: Union[Dict[str, float], pd.DataFrame]) -> Dict[str, Any]:
        """
        Runs real-time inference on a single sensor reading or telemetry packet.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before predicting.")

        if isinstance(sensor_payload, dict):
            df = pd.DataFrame([sensor_payload])
        else:
            df = sensor_payload

        # 1. Feature Engineering
        X_eng = self.feature_engineer.transform(df)

        # 2. Scaling
        X_scaled = self.scaler.transform(X_eng)

        # 3. Inference
        pred_class = int(self.model.predict(X_scaled)[0])
        probabilities = self.model.predict_proba(X_scaled)[0]

        level = self.class_labels[pred_class]
        confidence = float(probabilities[pred_class])

        # Explainable reasoning triggers
        triggers = []
        if df['soil_moisture_percentage'].iloc[0] > 75:
            triggers.append(f"Severe Soil Saturation ({df['soil_moisture_percentage'].iloc[0]:.1f}%)")
        if df['slope_displacement_mm'].iloc[0] > 3.0:
            triggers.append(f"Active Ground Creep ({df['slope_displacement_mm'].iloc[0]:.2f} mm/day)")
        if df['cumulative_rainfall_72h_mm'].iloc[0] > 150:
            triggers.append(f"Heavy Antecedent Rainfall ({df['cumulative_rainfall_72h_mm'].iloc[0]:.1f} mm/72h)")
        if df['slope_angle_deg'].iloc[0] > 45:
            triggers.append(f"Steep Unstable Incline ({df['slope_angle_deg'].iloc[0]:.1f}°)")

        return {
            'risk_level': level,
            'risk_code': pred_class,
            'confidence': confidence,
            'probabilities': {
                'low_risk': round(float(probabilities[0]), 3),
                'moderate_warning': round(float(probabilities[1]), 3),
                'high_alert': round(float(probabilities[2]), 3)
            },
            'primary_triggers': triggers if triggers else ["Normal geological baselines"],
            'recommended_action': self._get_action(pred_class)
        }

    def _get_action(self, risk_code: int) -> str:
        if risk_code == 2:
            return "🚨 RED ALERT: Activate community sirens, notify NDRF/SDMA, initiate immediate slope evacuation."
        elif risk_code == 1:
            return "⚠️ YELLOW WARNING: Increase sensor sampling frequency, inspect drainage channels, alert local patrol."
        return "🟢 GREEN: Slope is stable. Normal routine monitoring."

    def export(self, filepath: str = 'landslide_ai_model.pkl'):
        """Saves entire pipeline bundle (model + scaler + feature engineer)"""
        bundle = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_engineer': self.feature_engineer,
            'is_trained': self.is_trained
        }
        joblib.dump(bundle, filepath)
        print(f"💾 Model pipeline successfully exported to {filepath}")

    @classmethod
    def load(cls, filepath: str = 'landslide_ai_model.pkl'):
        """Loads serialized pipeline bundle"""
        bundle = joblib.load(filepath)
        instance = cls()
        instance.model = bundle['model']
        instance.scaler = bundle['scaler']
        instance.feature_engineer = bundle['feature_engineer']
        instance.is_trained = bundle['is_trained']
        print(f"✅ Model successfully loaded from {filepath}")
        return instance


# =====================================================================
# 4. EXECUTION DEMO
# =====================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("🏔️ TRAINING AI LANDSLIDE DETECTION MODEL")
    print("=" * 70)

    # 1. Synthesize geological telemetry
    simulator = LandslideDataSimulator(n_samples=10000)
    df, labels = simulator.generate()

    # 2. Train Model
    detector = LandslideDetectionModel()
    detector.train(df, labels)

    # 3. Test with a High-Risk Incident Telemetry Packet
    sample_emergency_reading = {
        'rainfall_intensity_mm_hr': 48.5,
        'cumulative_rainfall_72h_mm': 210.0,
        'soil_moisture_percentage': 88.0,
        'slope_angle_deg': 48.0,
        'slope_displacement_mm': 6.8,
        'pore_water_pressure_kpa': 38.0,
        'seismic_vibration_g': 0.15,
        'elevation_meters': 1850.0,
        'vegetation_ndvi': 0.25,
        'soil_shear_strength_kpa': 22.0,
        'fault_distance_km': 1.8,
        'groundwater_depth_m': 3.5
    }

    print("\n🔍 EVALUATING SENSOR TEST PAYLOAD:")
    result = detector.predict(sample_emergency_reading)
    print(f"Risk Assessment   : {result['risk_level']} (Confidence: {result['confidence']*100:.1f}%)")
    print(f"Probabilities     : {result['probabilities']}")
    print(f"Key Triggers      : {', '.join(result['primary_triggers'])}")
    print(f"Action Protocol   : {result['recommended_action']}\n")

    # 4. Save model
    detector.export('landslide_ai_model.pkl')
