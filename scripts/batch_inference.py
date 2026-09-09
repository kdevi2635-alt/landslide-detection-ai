"""
=============================================================================
Batch Inference Script
Process multiple sensor readings from CSV and generate predictions
=============================================================================
"""

import argparse
import pandas as pd
import time
from pathlib import Path
from landslide_detection_model import LandslideDetectionModel


def load_sensor_data(csv_path: str) -> pd.DataFrame:
    """Load sensor readings from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} sensor readings from {csv_path}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        raise
    except Exception as e:
        print(f"❌ Error loading CSV: {str(e)}")
        raise


def run_batch_inference(model: LandslideDetectionModel, 
                       data: pd.DataFrame) -> pd.DataFrame:
    """Run inference on all sensor readings"""
    
    predictions = []
    start_time = time.time()
    
    print(f"\n🔄 Running inference on {len(data)} samples...")
    
    for idx, row in data.iterrows():
        # Convert row to dict
        sensor_data = row.to_dict()
        
        # Predict
        result = model.predict(sensor_data)
        
        # Extract results
        pred_dict = {
            'sample_id': idx + 1,
            'risk_level': result['risk_level'],
            'risk_code': result['risk_code'],
            'confidence': round(result['confidence'], 4),
            'low_risk_prob': round(result['probabilities']['low_risk'], 4),
            'moderate_warning_prob': round(result['probabilities']['moderate_warning'], 4),
            'high_alert_prob': round(result['probabilities']['high_alert'], 4),
            'triggers_count': len(result['primary_triggers']),
            'primary_triggers': ' | '.join(result['primary_triggers'][:2])
        }
        
        predictions.append(pred_dict)
        
        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"   Processed {idx + 1}/{len(data)} samples...")
    
    inference_time = time.time() - start_time
    print(f"✅ Inference completed in {inference_time:.2f}s ({len(data)/inference_time:.0f} samples/sec)")
    
    return pd.DataFrame(predictions)


def generate_summary(predictions: pd.DataFrame) -> None:
    """Print prediction summary statistics"""
    
    print("\n" + "="*60)
    print("📊 BATCH PREDICTION SUMMARY")
    print("="*60)
    
    # Risk distribution
    risk_dist = predictions['risk_level'].value_counts()
    print(f"\n🎯 Risk Level Distribution:")
    for level, count in risk_dist.items():
        pct = (count / len(predictions)) * 100
        print(f"   {level:20s}: {count:5d} ({pct:5.1f}%)")
    
    # Confidence statistics
    print(f"\n📈 Confidence Statistics:")
    print(f"   Mean:     {predictions['confidence'].mean():.4f}")
    print(f"   Min:      {predictions['confidence'].min():.4f}")
    print(f"   Max:      {predictions['confidence'].max():.4f}")
    print(f"   Std Dev:  {predictions['confidence'].std():.4f}")
    
    # High risk alerts
    high_risk = predictions[predictions['risk_code'] == 2]
    print(f"\n🚨 High-Risk Samples: {len(high_risk)} ({(len(high_risk)/len(predictions))*100:.1f}%)")
    if len(high_risk) > 0:
        print(f"   Avg Confidence: {high_risk['confidence'].mean():.4f}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Run batch inference on sensor telemetry CSV"
    )
    parser.add_argument(
        "--data-path", 
        type=str, 
        required=True,
        help="Path to input CSV file with sensor readings"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="landslide_ai_model.pkl",
        help="Path to serialized model (default: landslide_ai_model.pkl)"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Path to output CSV"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("🏔️ BATCH INFERENCE - LANDSLIDE DETECTION AI")
    print("="*60)
    
    # Load model
    print(f"\n📦 Loading model from {args.model}...")
    try:
        detector = LandslideDetectionModel.load(args.model)
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        return
    
    # Load data
    data = load_sensor_data(args.data_path)
    
    # Run inference
    predictions = run_batch_inference(detector, data)
    
    # Generate summary
    generate_summary(predictions)
    
    print("\n✅ Batch inference complete!\n")


if __name__ == "__main__":
    main()
