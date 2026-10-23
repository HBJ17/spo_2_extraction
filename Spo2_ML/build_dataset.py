import os
import glob
import pandas as pd
from extract_features import process_video
import argparse

def build_dataset(video_dir, ground_truth_csv, output_csv, window_sec=5, step_sec=2):
    """
    Iterates over all videos, extracts windowed features, and links them to the ground truth SpO2.
    """
    if not os.path.exists(ground_truth_csv):
        print(f"Error: Ground truth file '{ground_truth_csv}' not found.")
        return
        
    # Read ground truth
    # Assuming CSV format: filename, spo2
    try:
        gt_df = pd.read_csv(ground_truth_csv)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return
        
    # Clean up column names for robustness (strip spaces, lowercase)
    gt_df.columns = gt_df.columns.str.strip().str.lower()
    
    # We expect 'filename' or 'video' and 'spo2' columns
    filename_col = next((col for col in gt_df.columns if 'file' in col or 'video' in col or 'name' in col), None)
    spo2_col = next((col for col in gt_df.columns if 'spo2' in col or 'truth' in col or 'value' in col), None)
    
    if not filename_col or not spo2_col:
        print(f"Could not find required columns in {ground_truth_csv}.")
        print(f"Found columns: {list(gt_df.columns)}. Expected something like 'filename' and 'spo2'.")
        return
        
    # Create a mapping from filename (without extension) to SpO2 value
    gt_mapping = {}
    for _, row in gt_df.iterrows():
        base_name = os.path.splitext(str(row[filename_col]).strip())[0]
        gt_mapping[base_name] = float(row[spo2_col])
        
    print(f"Loaded ground truth for {len(gt_mapping)} videos.")
    
    video_files = glob.glob(os.path.join(video_dir, "*.mp4")) + glob.glob(os.path.join(video_dir, "*.avi")) + glob.glob(os.path.join(video_dir, "*.mov"))
    print(f"Found {len(video_files)} video files in {video_dir}.")
    
    all_data = []
    
    for i, video_path in enumerate(video_files):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        if base_name not in gt_mapping:
            print(f"Warning: No ground truth for {base_name}. Skipping.")
            continue
            
        true_spo2 = gt_mapping[base_name]
        
        print(f"[{i+1}/{len(video_files)}] Processing {os.path.basename(video_path)} (SpO2: {true_spo2})...")
        
        # Extract features for all windows
        features_list = process_video(video_path, window_sec=window_sec, step_sec=step_sec)
        
        if not features_list:
            print(f"  -> Failed to extract features or video empty.")
            continue
            
        # Append ground truth and subject ID to each window feature set
        for window_idx, feat in enumerate(features_list):
            feat['subject_id'] = base_name  # Important for GroupKFold Cross-Validation
            feat['window_idx'] = window_idx
            feat['true_spo2'] = true_spo2
            all_data.append(feat)
            
        print(f"  -> Extracted {len(features_list)} windows.")
        
    if not all_data:
        print("No data extracted. Check your video paths and ground truth mapping.")
        return
        
    # Convert to DataFrame and save
    final_df = pd.DataFrame(all_data)
    final_df.to_csv(output_csv, index=False)
    print(f"\nSuccess! Dataset built with {len(final_df)} total samples (windows) from {len(final_df['subject_id'].unique())} videos.")
    print(f"Saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SpO2 ML Dataset")
    parser.add_argument("--video_dir", type=str, default=".", help="Directory containing video files")
    parser.add_argument("--gt_csv", type=str, default="ground_truth.csv", help="Path to ground truth CSV file")
    parser.add_argument("--out_csv", type=str, default="dataset.csv", help="Output dataset CSV file")
    parser.add_argument("--window", type=int, default=5, help="Window size in seconds")
    parser.add_argument("--step", type=int, default=2, help="Step size in seconds")
    
    args = parser.parse_args()
    build_dataset(args.video_dir, args.gt_csv, args.out_csv, args.window, args.step)
