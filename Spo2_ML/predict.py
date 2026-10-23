import argparse
import pickle
import numpy as np
import pandas as pd
from extract_features import process_video
import warnings
warnings.filterwarnings('ignore') # Ignore sklearn warnings for cleaner output

def predict_video(video_path, model_path="best_spo2_model.pkl"):
    print(f"\nProcessing video: {video_path}")
    print("Extracting features (this might take a few seconds)...")
    
    # 1. Extract 5-second windows from the video
    windows = process_video(video_path, window_sec=5, step_sec=2)
    
    if not windows:
        print("Error: Could not extract any valid features. Is the video empty?")
        return
        
    print(f"Successfully extracted {len(windows)} overlapping windows.")
    
    # 2. Convert to DataFrame to easily compute formula and match feature columns
    df = pd.DataFrame(windows)
    
    # Calculate the formula baseline feature
    df['formula_spo2'] = 110 - 25 * df['RoR_g']
    
    # 3. Load the trained model, scaler, and expected feature list
    try:
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            model = saved_data['model']
            scaler = saved_data['scaler']
            expected_features = saved_data['features']
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found! Please run train_model.py first.")
        return
        
    # Ensure our extracted features match what the model expects
    X_raw = df[expected_features].values
    
    # 4. Scale features
    X_scaled = scaler.transform(X_raw)
    
    # 5. Predict for every single window
    window_predictions = model.predict(X_scaled)
    
    for i, pred in enumerate(window_predictions):
        print(f"  -> Window {i+1} prediction: {pred:.1f}%")
        
    # 6. Apply Video-Level Aggregation (Median Filter)
    final_prediction = np.round(np.median(window_predictions))
    
    print(f" FINAL PREDICTED SpO2: {final_prediction:.0f}% ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SpO2 ML Model on a Video")
    parser.add_argument("video", type=str, help="Path to the video file (e.g., 'D:\\Studies\\ML\\DATA\\Video\\2510017.mp4')")
    parser.add_argument("--model", type=str, default="best_spo2_model.pkl", help="Path to saved model")
    
    args = parser.parse_args()
    predict_video(args.video, args.model)
